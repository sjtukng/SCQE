"""Measure D true-risk and coverage scales for B1/B2/B3 without VJP."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR
from scqe.coverage import Coverage
from scqe.data import support_instances
from scqe.pf_runtime import PFRuntime
from scqe.pipeline import QueryEngine
from scqe.risk import true_risk
from scqe.solver import netgreedy
from scqe.support import SupportEncoder


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a = json.loads(args.acceptance.read_text())
    if len(a.get("forward_checks", [])) != 3 or \
            not all(x.get("pass_forward") for x in a["forward_checks"]) or \
            not a.get("synthetic_cache_test") or \
            len(a.get("solver_checks", [])) != 3 or \
            not all(x.get("passed") for x in a["solver_checks"]):
        raise RuntimeError("B1/B2/B3 forward/cache/solver acceptance has not passed")
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; CPU fallback disabled")
    entries = json.loads(args.split.read_text())["queries"]
    if len(entries) != 40:
        raise ValueError("expected frozen Dcal40 manifest")
    args.out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rt = PFRuntime(model, "cuda:0")
    encoder = SupportEncoder(rt)
    supports = {seed: encoder.encode_episode(support_instances(seed, 5), 20)
                for seed in (2023, 42)}
    records, true_positive, true_noise, coverage_repeat = [], [], [], []
    for entry in entries[:12]:
        qid = str(entry["image_id"])
        engine = QueryEngine(rt, "DIOR", qid).prepare(
            str(DIOR_DIR / "images" / entry["image_path"]))
        for seed in (2023, 42):
            p, t, valid = supports[seed]
            alpha, cols, status = engine.build_alpha(p, t, valid)
            rec = dict(image_id=qid, support_seed=seed, status=status, planned_m=0)
            if alpha is None:
                records.append(rec)
                continue
            ctx = engine.ctx
            cov1 = Coverage(ctx.vs_idx, ctx.vs_boxes, alpha, cols, ctx.A0)
            cov2 = Coverage(ctx.vs_idx, ctx.vs_boxes, alpha, cols, ctx.A0)
            coverage_repeat.append(abs(cov1.F() - cov2.F()))
            out = netgreedy(ctx, alpha, cols, None, m_max=32, delta_F=0.0)
            rec.update(planned_m=out["m"], F0=out["F0"], Fplan=out["Fplan"])
            if out["m"]:
                values = [rt.native_decode_R(engine.state, engine.R_of_plan(out["plan"]))[0]
                          for _ in range(3)]
                risks = [true_risk(y, engine.state.y0, engine.state.wsqrt)
                         for y in values]
                noise = max(risks) - min(risks)
                true_noise.append(noise)
                rec.update(true_risk=risks[0], true_repeat_range=noise)
                if risks[0] > 0:
                    true_positive.append(risks[0])
            records.append(rec)
    noise_floor = 10 * max(true_noise, default=0.0)
    positive = [x for x in true_positive if x > noise_floor]
    if not positive:
        raise RuntimeError("no positive true-risk calibration scale above noise")
    delta_f = max(10 * max(coverage_repeat, default=0.0),
                  10 * torch.finfo(torch.float32).eps)
    data = dict(protocol="D", dataset="DIOR", shot=5,
                role="non-derivative B1/B2/B3 calibration only",
                status="MEASURED", images=12, support_seeds=[2023, 42],
                records=records, s_t=statistics.median(positive),
                true_positive_n=len(positive), true_noise_floor=noise_floor,
                delta_F=delta_f, split_sha256=digest(args.split),
                acceptance_sha256=digest(args.acceptance),
                pf_checkpoint_sha256=digest(CHECKPOINT_PATH),
                caveat="SCQE derivative E2 remains failed and is excluded")
    (args.out / "scales.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
