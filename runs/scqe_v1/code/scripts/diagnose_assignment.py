"""E7 offline fixed-member assignments; never selects a deployed prediction."""
import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR
from scqe.coverage import Coverage
from scqe.pf_runtime import PFRuntime
from scqe.risk import true_risk


def proxy_values(g, R0, xi, A0, slots, plan):
    delta = (xi[plan] - R0).to(torch.float64)
    projections = torch.einsum("prc,rc->p", g.to(torch.float64), delta)
    cumulative = float(projections.square().mean())
    isolated = 0.0
    for slot in slots:
        d = (g[:, slot].to(torch.float64) * delta[slot]).sum(-1)
        isolated += float(d.square().mean())
    return cumulative, isolated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; no CPU model fallback")
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    if len(entries) != 20:
        raise ValueError("assignment diagnostic needs frozen 20-query subset")
    meta = json.loads((args.run / "metadata.json").read_text())
    launch = json.loads((args.run / "launch.json").read_text())
    if meta["seed"] != 2023 or meta["shot"] != 5 or meta["expected_queries"] != 20:
        raise ValueError("wrong diagnostic episode")
    if launch["status"] != "COMPLETE" or \
            meta["split_sha256"] != hashlib.sha256(args.split.read_bytes()).hexdigest() or \
            meta["lock_sha256"] != hashlib.sha256(args.lock.read_bytes()).hexdigest():
        raise ValueError("diagnostic run, frozen split and parameter lock differ")
    lock = json.loads(args.lock.read_text())
    eps = lock["selections"]["SCQE"]["eps_true"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rt = PFRuntime(model, "cuda:0")
    rows = []
    for entry in entries:
        qid = str(entry["image_id"])
        record = json.loads((args.run / "plans" / f"{qid}__SCQE.json").read_text())
        if len(record["steps"]) < 2 or record["status"] not in ("accepted", "true_reject"):
            continue
        saved = torch.load(args.run / "states" / f"{qid}.pt", map_location="cuda:0")
        alpha, g = saved["alpha"], saved["g"]
        if alpha is None or g is None:
            raise RuntimeError(f"missing re-computation state: {qid}")
        st = rt.native_pass(str(DIOR_DIR / "images" / entry["image_path"]))
        if not torch.equal(st.A0, saved["A0"]) or not torch.allclose(st.R0, saved["R0"], atol=0, rtol=0):
            raise RuntimeError(f"baseline changed since recorded run: {qid}")
        slots = sorted(step["slot"] for step in record["steps"])
        incoming = [step["incoming"] for step in record["steps"]]
        lookup = torch.full((len(st.xi_all),), -1, dtype=torch.long, device="cuda:0")
        lookup[saved["vs_idx"]] = torch.arange(len(saved["vs_idx"]), device="cuda:0")
        ranked = sorted(incoming, key=lambda j: (-float(alpha[lookup[j]].max()), j))
        original = tuple(int(record["Astar"][slot]) for slot in slots)
        canonical = tuple(ranked)
        rng = random.Random(20261014 + int(qid))
        permutations = []
        unique = list(set(itertools.permutations(ranked))) if len(ranked) <= 7 else None
        if unique is not None:
            unique = [perm for perm in unique if perm != canonical]
            rng.shuffle(unique)
            permutations = unique[:5]
        else:
            seen = {canonical}
            for _ in range(500):
                candidate = tuple(rng.sample(ranked, len(ranked)))
                if candidate not in seen:
                    permutations.append(candidate)
                    seen.add(candidate)
                if len(permutations) == 5:
                    break
        assignments = [("original", original), ("canonical", canonical)] + [
            (f"random_{i+1}", perm) for i, perm in enumerate(permutations)]
        F_values = []
        if record["true_risk"] is None:
            raise RuntimeError(f"missing original pre-gate risk: {qid}")
        baseline_gate = record["true_risk"] <= eps
        for label, perm in assignments:
            plan = st.A0.clone()
            for slot, candidate in zip(slots, perm):
                plan[slot] = candidate
            F = Coverage(saved["vs_idx"], saved["vs_boxes"], alpha,
                         list(range(alpha.shape[1])), plan).F()
            F_values.append(F)
            cumulative, isolated = proxy_values(g, st.R0, st.xi_all, st.A0, slots, plan)
            y, _, _ = rt.native_decode_R(st, st.xi_all[plan].unsqueeze(0))
            actual = true_risk(y, st.y0, st.wsqrt)
            rows.append(dict(image_id=int(qid), assignment=label,
                             permutation=list(perm), slots=slots, F=F,
                             proxy_cumulative=cumulative, proxy_isolated=isolated,
                             true_risk=actual, accepted_under_locked_gate=actual <= eps,
                             original_true_risk=record["true_risk"],
                             true_risk_delta=actual - record["true_risk"],
                             gate_flip=(actual <= eps) != baseline_gate,
                             original_status=record["status"],
                             random_permutations=len(permutations),
                             offline_extra_forward=True))
        if max(F_values) - min(F_values) > 1e-8:
            raise RuntimeError(f"fixed-member F changed under permutation: {qid}")
    if not rows:
        raise RuntimeError("no accepted/rejected plans with at least two exchanges")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
