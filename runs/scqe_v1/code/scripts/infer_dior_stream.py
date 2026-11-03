"""PF-RPN/SCQE producer for one frozen DIOR episode and query manifest."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR, SEED_GLOBAL
from scqe.data import support_instances
from scqe.pf_runtime import PFRuntime
from scqe.pipeline import QueryEngine
from scqe.support import SupportEncoder


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w") as f:
        json.dump(value, f, allow_nan=False)
    os.replace(temp, path)


def as_json(res):
    keep = ("method", "status", "planned_m", "accepted_m", "F0", "Fplan",
            "true_risk", "steps", "proxy_cum", "proxy_isolated", "probe_seed",
            "fallback_verified", "reason", "prefix_info", "risk_groups")
    rec = {k: res[k] for k in keep if k in res}
    for k in ("A0", "Astar", "Aout"):
        rec[k] = res[k].tolist()
    return rec


def params(method, selections, delta):
    cfg = selections.get(method, selections.get("SCQE", {}))
    if method in ("B1", "B2"):
        return dict(budget=int(cfg["budget"]), delta_F=0.0)
    if method == "B3":
        return dict(eps_true=float(cfg["eps_true"]), delta_F=0.0)
    if method in ("SCQE", "A1", "A2", "A3", "A4", "A5", "A3-Cal"):
        return dict(eps_proxy=float(cfg["eps_proxy"]),
                    eps_true=float(cfg["eps_true"]), delta_F=delta)
    return {}


def a1_diagnostics(eng, p, t, valid, alpha, result):
    """Record object/context cosines and alpha masses for NoContext."""
    if eng.ctx is None or alpha is None:
        return None
    z = eng.ctx.z_vs
    object_cos = z @ p.T
    context_cos = z @ t.T
    positive = torch.nonzero(valid).flatten()
    class_rows = []
    for c in positive.tolist():
        evidence = ((object_cos[:, c] / 2).clamp(0, 1))
        class_rows.append(dict(class_id=c, object_cos_mean=float(object_cos[:, c].mean()),
                               context_cos_mean=float(context_cos[:, c].mean()),
                               alpha_mass=float(evidence.sum()),
                               alpha_positive=int((evidence > 0).sum())))
    lookup = torch.full((len(eng.state.xi_all),), -1, dtype=torch.long, device=z.device)
    lookup[eng.ctx.vs_idx] = torch.arange(len(z), device=z.device)
    selected = []
    for step in result["steps"]:
        j = step["incoming"]
        pos = int(lookup[j])
        if pos >= 0:
            selected.append(dict(incoming=j, slot=step["slot"],
                                 object_cos=object_cos[pos, positive].tolist(),
                                 context_cos=context_cos[pos, positive].tolist(),
                                 alpha=(object_cos[pos, positive] / 2).clamp(0, 1).tolist()))
    return dict(classes=positive.tolist(), per_class=class_rows, selected_incoming=selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--methods", default="B0,B1,B2,B3,SCQE")
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--spool", type=Path, required=True)
    ap.add_argument("--record-state", action="store_true")
    ap.add_argument("--probes", type=int, choices=(4, 16, 32), default=16)
    ap.add_argument("--budget-override", type=int, choices=(8, 16, 32))
    ap.add_argument("--backend-device", choices=("cuda:0", "cuda:1"), default="cuda:1")
    ap.add_argument("--resume-prefix", type=int, default=0,
                    help="number of complete leading queries copied into this new spool")
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("a CUDA GPU is required; CPU fallback is disabled")
    if args.resume_prefix < 0:
        raise ValueError("resume prefix must be nonnegative")
    staged = args.resume_prefix > 0
    if (args.spool / "metadata.json").exists() or \
            ((args.spool / "jobs").exists() and not staged):
        raise FileExistsError("spool already contains a run")
    args.spool.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "plans", "states", "planned_proposals"):
        (args.spool / name).mkdir(exist_ok=staged)
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = json.loads(args.lock.read_text())
    if lock.get("protocol") != "D" or lock.get("dataset") != "DIOR":
        raise ValueError("DIOR D protocol requires its own completed lock")
    delta = float(lock["delta_F"])
    selections = lock["selections"]
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    if args.resume_prefix > len(entries):
        raise ValueError("resume prefix exceeds query count")
    if staged:
        manifest = json.loads((args.spool / "resume_manifest.json").read_text())
        if manifest.get("prefix_queries") != args.resume_prefix or \
                manifest.get("split_sha256") != sha(args.split):
            raise RuntimeError("staged resume manifest does not match full split")
    methods = args.methods.split(",")
    if len(methods) != len(set(methods)):
        raise ValueError("duplicate methods")
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rt = PFRuntime(model, device="cuda:0")
    support_path = Path(__file__).resolve().parents[3] / "sqr_v1" / "splits" / \
        f"dior_support_{args.shot}shot_seed{args.seed}.json"
    instances = support_instances(args.seed, args.shot)
    if len(instances) != 20 * args.shot:
        raise ValueError("support annotation budget differs from 20 x shot")
    baseline_only = methods == ["B0"]
    p, t, valid = (None, None, None) if baseline_only else \
        SupportEncoder(rt).encode_episode(instances, 20)
    backend = Path(__file__).resolve().parents[2] / "backend" / \
        f"seed{args.seed}_shot{args.shot}"
    backend_checkpoint = backend / "adapt/model_final.pth"
    backend_prototype = backend / \
        f"DIOR_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    for path in (backend_checkpoint, backend_prototype):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = dict(protocol="D", dataset="DIOR", shot=args.shot, seed=args.seed,
                    split_sha256=sha(args.split), support_sha256=sha(support_path),
                    lock_sha256=sha(args.lock), checkpoint_sha256=sha(CHECKPOINT_PATH),
                    backend_checkpoint_sha256=sha(backend_checkpoint),
                    backend_prototype_sha256=sha(backend_prototype),
                    methods=methods, expected_queries=len(entries),
                    probes=args.probes, budget_override=args.budget_override,
                    gpu_proposal="cuda:0", gpu_backend=args.backend_device,
                    execution_mode="sequential_single_gpu" if args.backend_device == "cuda:0"
                    else "concurrent_two_gpu")
    atomic_json(args.spool / "metadata.json", metadata)
    for index in range(args.resume_prefix, len(entries)):
        entry = entries[index]
        qid = str(entry["image_id"])
        image_path = str(DIOR_DIR / "images" / entry["image_path"])
        torch.cuda.synchronize(0)
        prep_start = time.perf_counter_ns()
        eng = QueryEngine(rt, "DIOR", qid, probes=args.probes)
        eng = eng.prepare_baseline(image_path) if baseline_only else eng.prepare(image_path)
        torch.cuda.synchronize(0)
        prep_ms = (time.perf_counter_ns() - prep_start) / 1e6
        alpha, cols = None, []
        torch.cuda.synchronize(0)
        evidence_start = time.perf_counter_ns()
        if not eng.bypass_reason and any(m != "B0" for m in methods):
            alpha, cols, _ = eng.build_alpha(p, t, valid)
        alpha_no_context, cols_no_context = None, []
        if "A1" in methods and not eng.bypass_reason:
            alpha_no_context, cols_no_context, _ = eng.build_alpha(p, t, valid, no_context=True)
        torch.cuda.synchronize(0)
        evidence_ms = (time.perf_counter_ns() - evidence_start) / 1e6
        method_results = {}
        for method in methods:
            a = alpha_no_context if method == "A1" else alpha
            c = cols_no_context if method == "A1" else cols
            method_params = params(method, selections, delta)
            if args.budget_override is not None and method in ("SCQE", "A1", "A2",
                                                                  "A3", "A4", "A5", "A3-Cal"):
                method_params["budget"] = args.budget_override
            torch.cuda.synchronize(0)
            method_start = time.perf_counter_ns()
            result = eng.run_method(method, a, c, **method_params)
            method_results[method] = result
            torch.cuda.synchronize(0)
            method_ms = (time.perf_counter_ns() - method_start) / 1e6
            record = as_json(result)
            record.update(image_id=int(qid), image_path=image_path, query_index=index,
                          shared_preprocess_encoder_ms=prep_ms,
                          shared_evidence_ms=evidence_ms, method_gpu0_ms=method_ms)
            if method == "A1":
                record["no_context_diagnostics"] = a1_diagnostics(
                    eng, p, t, valid, a, result)
            atomic_json(args.spool / "plans" / f"{qid}__{method}.json", record)
            if "planned_boxes" in result:
                atomic_json(args.spool / "planned_proposals" / f"{qid}__{method}.json",
                            dict(boxes=result["planned_boxes"].tolist(),
                                 scores=result["planned_scores"].tolist()))
            box_list = result["boxes"].tolist()
            score_list = result["scores"].tolist()
            atomic_json(args.spool / "jobs" / f"{qid}__{method}.ready.json",
                        dict(image_id=int(qid), file_name=image_path,
                             height=int(entry["height"]), width=int(entry["width"]),
                             method=method, boxes=box_list, scores=score_list))
        if "SCQE" in method_results and "A4" in method_results:
            full, no_gate = method_results["SCQE"], method_results["A4"]
            if not torch.equal(full["Astar"], no_gate["Astar"]) or \
                    full["steps"] != no_gate["steps"]:
                raise RuntimeError(f"A4/SCQE pre-gate plans differ for image {qid}")
        if args.record_state:
            torch.save(dict(R0=eng.state.R0, A0=eng.state.A0, H=eng.state.H_slots,
                            y0=eng.state.y0, Wsqrt=eng.state.wsqrt,
                            img_W=eng.state.img_W, img_H=eng.state.img_H,
                            scale_factor=eng.state.scale_factor,
                            xi_all=eng.state.xi_all,
                            vs_idx=eng.ctx.vs_idx if eng.ctx else None,
                            vs_boxes=eng.ctx.vs_boxes if eng.ctx else None,
                            alpha=alpha, alpha_no_context=alpha_no_context,
                            g=eng.g, p=p, t=t, class_valid=valid),
                       args.spool / "states" / f"{qid}.pt")
        print(f"[{index + 1}/{len(entries)}] {qid}", flush=True)
    atomic_json(args.spool / "producer_done.json",
                dict(status="COMPLETE", count=len(entries),
                     resumed_prefix_queries=args.resume_prefix,
                     no_risk_search_cache_verified=QueryEngine._no_risk_cache_checks))


if __name__ == "__main__":
    main()
