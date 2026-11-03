"""Formal online latency measurement (spec 11.1, pilot subset).

50 fixed D_check queries x one 5-shot support (fresh query-side compute per
method; support adapted+cached). 5 warmup + 3 measured rounds, alternating
method order. Records stage-level wall times and peak memory for B0 and M.

Note: classifier runs on GPU1 in production settings; here we measure the
PF/VJP/solver/gate side (GPU0) per method and classify separately.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, M_MAX, SEED_GLOBAL
from scqe.data import dcheck_entries, query_image_path, support_instances
from scqe.pipeline import QueryEngine
from scqe.solver import netgreedy
from scqe.support import SupportEncoder

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
OUT = ROOT / "runs/scqe_v1/e3_pilot"


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--device", default="cuda:0")
    ap_.add_argument("--n", type=int, default=50)
    ap_.add_argument("--seed", type=int, default=2023)
    args = ap_.parse_args()
    torch.manual_seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    frozen = json.load(open(ROOT / "runs/scqe_v1/e2_calibration/e2_frozen_config.json"))
    sel = frozen["selections"]
    eps_p, eps_t = sel["M"]["eps_proxy"], sel["M"]["eps_true"]

    from mmdet.apis import init_detector
    from scqe.pf_runtime import PFRuntime
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    model.eval()
    rt = PFRuntime(model, device=args.device)
    enc = SupportEncoder(rt)
    inst = support_instances(args.seed, 5)
    p, t, c_valid = enc.encode_episode(inst, n_classes=20)

    entries = dcheck_entries()[: args.n]
    imgs = [query_image_path(e) for e in entries]

    rows = []

    def run_one(method, img, qid):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        eng = QueryEngine(rt, "DIOR", qid).prepare(img)
        t_prep = time.time() - t0
        if eng.bypass_reason:
            return dict(method=method, qid=qid, status=f"bypass_{eng.bypass_reason}")
        alpha, cols, _ = eng.build_alpha(p, t, c_valid)
        st = {}
        if method == "M":
            t0 = time.time()
            out = netgreedy(eng.ctx, alpha, cols, eng.g, m_max=M_MAX,
                            eps_proxy=eps_p)
            st["solver_s"] = time.time() - t0
            st["planned_m"] = out["m"]
            t0 = time.time()
            if out["m"] > 0:
                from scqe.risk import true_risk
                y, s_pf, extras = rt.native_decode_R(eng.state,
                                                     eng.R_of_plan(out["plan"]))
                st["changed_decoder_s"] = time.time() - t0
                st["true_risk"] = true_risk(y, eng.state.y0, eng.state.wsqrt)
            else:
                st["changed_decoder_s"] = 0.0
        row = dict(method=method, qid=qid, prep_s=t_prep, **st,
                   total_s=time.time() - (time.time() - t_prep),
                   peak_mb=torch.cuda.max_memory_allocated() / 1e6)
        return row

    methods = ["B0", "M"]
    for w in range(5):  # warmup
        for img, e in zip(imgs, entries):
            run_one(methods[w % 2], img, e["image_id"])
    print("warmup done", flush=True)
    for rnd in range(3):
        order = methods if rnd % 2 == 0 else methods[::-1]
        for mth in order:
            for img, e in zip(imgs, entries):
                rows.append(run_one(mth, img, e["image_id"]))
        print(f"round {rnd+1} done", flush=True)

    with open(OUT / "cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for mth in methods:
        ts = [r["prep_s"] for r in rows if r["method"] == mth and "prep_s" in r]
        peaks = [r["peak_mb"] for r in rows if r["method"] == mth and "peak_mb" in r]
        print(f"{mth}: mean={np.mean(ts):.3f}s p50={np.percentile(ts,50):.3f} "
              f"p95={np.percentile(ts,95):.3f} peak={max(peaks):.0f}MB")
    m_rows = [r for r in rows if r["method"] == "M" and "solver_s" in r]
    print(f"M solver mean={np.mean([r['solver_s'] for r in m_rows]):.3f}s "
          f"changed-decoder mean={np.mean([r['changed_decoder_s'] for r in m_rows]):.3f}s")


if __name__ == "__main__":
    main()
