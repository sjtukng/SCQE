"""E2 calibration: D_cal(40) x 5-shot x supports {2023,42}.

- B1 budgets {8,16,32}, B2 budgets {8,16,32} (independent selection on D_cal)
- B3: independent 32-seq + prefix true-risk replay at 3 true thresholds
- M: proxy-grid {s_p/3,s_p,3s_p} x true-grid {min(s_t/3,1),min(s_t,1),min(3s_t,1)}
Selection: mean paired AP (vs B0) over the 2 support episodes; ties -> smaller
eps_true then smaller eps_proxy (M); budget/threshold selected independently.
Saves proposals jsons + calibration.csv + metrics per (method, config).
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

from scqe.config import (CHECKPOINT_PATH, CONFIG_PATH, M_MAX, SEED_GLOBAL,
                         SEED_SUPPORT_CAL)
from scqe.data import dcal_entries, query_image_path, support_instances
from scqe.pipeline import QueryEngine
from scqe.risk import true_risk
from scqe.solver import b1_independent_rank, netgreedy
from scqe.support import SupportEncoder

OUT = Path(__file__).resolve().parents[2] / "e2_calibration"
OUT.mkdir(parents=True, exist_ok=True)


def decode_plan(rt, eng, plan):
    if plan is None or int((plan != eng.state.A0).sum()) == 0:
        return eng.state.b0_boxes, eng.state.b0_scores, 0.0
    y, s_pf, extras = rt.native_decode_R(eng.state, eng.R_of_plan(plan))
    tr = true_risk(y, eng.state.y0, eng.state.wsqrt)
    boxes, scores = rt.proposals_from_outputs(eng.state, extras[0], s_pf)
    return boxes, scores, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from mmdet.apis import init_detector
    from scqe.pf_runtime import PFRuntime
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    model.eval()
    rt = PFRuntime(model, device=args.device)
    enc = SupportEncoder(rt)
    supports = {}
    for seed in SEED_SUPPORT_CAL:
        inst = support_instances(seed, 5)
        p, t, c_valid = enc.encode_episode(inst, n_classes=20)
        supports[seed] = dict(p=p, t=t, c_valid=c_valid)
        print(f"support {seed}: {int(c_valid.sum())}/20 classes valid", flush=True)

    scales = json.load(open(OUT.parent / "e1_validation" / "scales.json"))
    s_p, s_t = scales["s_p"], scales["s_t"]
    assert s_p and s_t, "scales undefined; run E1 diagnostics first"
    proxy_grid = sorted({s_p / 3, s_p, 3 * s_p})
    true_grid = sorted({min(s_t / 3, 1), min(s_t, 1), min(3 * s_t, 1)})
    print("proxy_grid", proxy_grid, "true_grid", true_grid, flush=True)

    entries = dcal_entries()
    records = []
    proposals = {}   # (seed, run_key) -> {qid: {boxes, scores}}

    for qi, entry in enumerate(entries):
        qid = entry["image_id"]
        eng = QueryEngine(rt, "DIOR", qid).prepare(query_image_path(entry))
        for seed in SEED_SUPPORT_CAL:
            sp = supports[seed]
            if eng.bypass_reason:
                records.append(dict(qid=qid, seed=seed, method="B0",
                                    status=f"bypass_{eng.bypass_reason}"))
                continue
            alpha, cols, _ = eng.build_alpha(sp["p"], sp["t"], sp["c_valid"])
            if alpha is None:
                records.append(dict(qid=qid, seed=seed, method="B0",
                                    status="no_evidence"))
                continue
            pk0 = f"{seed}/B0"
            proposals.setdefault(pk0, {})[str(qid)] = dict(
                boxes=eng.state.b0_boxes.cpu().numpy().tolist(),
                scores=eng.state.b0_scores.cpu().numpy().tolist())
            records.append(dict(qid=qid, seed=seed, method="B0", status="ok",
                                planned_m=0, accepted_m=0))

            # B1 / B2 budgets
            for method in ("B1", "B2"):
                for budget in (8, 16, 32):
                    if method == "B1":
                        out = b1_independent_rank(eng.ctx, alpha, cols, budget)
                    else:
                        out = netgreedy(eng.ctx, alpha, cols, None, m_max=budget)
                    boxes, scores, tr = decode_plan(rt, eng, out["plan"])
                    pk = f"{seed}/{method}{budget}"
                    proposals.setdefault(pk, {})[str(qid)] = dict(
                        boxes=boxes.cpu().numpy().tolist(),
                        scores=scores.cpu().numpy().tolist())
                    records.append(dict(qid=qid, seed=seed, method=method,
                                        budget=budget, status=out["status"],
                                        planned_m=out["m"], F0=out["F0"],
                                        Fplan=out["Fplan"], true_risk=tr))

            # B3: sequence + prefix forwards
            out = netgreedy(eng.ctx, alpha, cols, None, m_max=M_MAX)
            k = out["m"]
            prefix_plans = {}
            if k > 0:
                cum = eng.state.A0.clone()
                step_of = {i + 1: s for i, s in enumerate(out["steps"])}
                for m in sorted({min(v, k) for v in (1, 4, 8, 16, 32)} - {0}):
                    srec = step_of[m]
                    cum = cum.clone()
                    cum[srec["slot"]] = srec["incoming"]
                    boxes, scores, tr = decode_plan(rt, eng, cum)
                    prefix_plans[m] = dict(plan=cum.cpu().tolist(), true_risk=tr)
                    pk = f"{seed}/B3_prefix{m}"
                    proposals.setdefault(pk, {})[str(qid)] = dict(
                        boxes=boxes.cpu().numpy().tolist(),
                        scores=scores.cpu().numpy().tolist())
            records.append(dict(qid=qid, seed=seed, method="B3",
                                status=out["status"], planned_m=k,
                                F0=out["F0"], Fplan=out["Fplan"],
                                prefix_true_risk=json.dumps(
                                    {m: prefix_plans[m]["true_risk"]
                                     for m in prefix_plans})))

            # M: 3 proxy plans (replayed against 3 true thresholds offline)
            m_plans = {}
            for eps_p in proxy_grid:
                outm = netgreedy(eng.ctx, alpha, cols, eng.g, m_max=M_MAX,
                                 eps_proxy=eps_p)
                if outm["m"] > 0:
                    boxes, scores, tr = decode_plan(rt, eng, outm["plan"])
                    for eps_t in true_grid:
                        accepted = tr <= eps_t
                        pk = f"{seed}/M_p{eps_p:.3e}_t{eps_t:.3e}"
                        if accepted:
                            proposals.setdefault(pk, {})[str(qid)] = dict(
                                boxes=boxes.cpu().numpy().tolist(),
                                scores=scores.cpu().numpy().tolist())
                        else:
                            proposals.setdefault(pk, {})[str(qid)] = dict(
                                boxes=eng.state.b0_boxes.cpu().numpy().tolist(),
                                scores=eng.state.b0_scores.cpu().numpy().tolist())
                    m_plans[f"{eps_p:.3e}"] = dict(
                        m=outm["m"], status=outm["status"], true_risk=tr,
                        F0=outm["F0"], Fplan=outm["Fplan"])
                else:
                    for eps_t in true_grid:
                        pk = f"{seed}/M_p{eps_p:.3e}_t{eps_t:.3e}"
                        proposals.setdefault(pk, {})[str(qid)] = dict(
                            boxes=eng.state.b0_boxes.cpu().numpy().tolist(),
                            scores=eng.state.b0_scores.cpu().numpy().tolist())
                    m_plans[f"{eps_p:.3e}"] = dict(m=0, status=out["status"])
            records.append(dict(qid=qid, seed=seed, method="M",
                                m_plans=json.dumps(m_plans)))
        print(f"[{qi+1}/{len(entries)}] {qid}", flush=True)

    with open(OUT / "calibration_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        w.writeheader()
        for r in records:
            w.writerow(r)
    for pk, val in proposals.items():
        p = OUT / "proposals" / f"{pk.replace('/', '__')}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(val, open(p, "w"))
    json.dump(dict(proxy_grid=proxy_grid, true_grid=true_grid,
                   s_p=s_p, s_t=s_t), open(OUT / "grid.json", "w"), indent=1)
    print("E2 stage-1 done: records + proposals saved")


if __name__ == "__main__":
    main()
