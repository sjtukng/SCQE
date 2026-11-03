"""E3 pilot: D_check(60) x 5-shot x support seeds {2023,42,1234} x methods
{B0,B1,B2,B3,M} with the frozen E2 config. No parameter adjustment.

Outputs per_query.csv, proposals/ and detections/ under e3_pilot/, plus
per-(method,seed) AP vs dcheck60 GT.
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
                         SEED_SUPPORT_PILOT)
from scqe.data import dcheck_entries, query_image_path, support_instances
from scqe.pipeline import QueryEngine
from scqe.risk import true_risk
from scqe.solver import b1_independent_rank, netgreedy
from scqe.support import SupportEncoder

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
OUT = ROOT / "runs/scqe_v1/e3_pilot"
(OUT / "proposals").mkdir(parents=True, exist_ok=True)
(OUT / "detections").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    frozen = json.load(open(ROOT / "runs/scqe_v1/e2_calibration/e2_frozen_config.json"))
    sel = frozen["selections"]
    eps_p_m = sel["M"]["eps_proxy"]
    eps_t_m = sel["M"]["eps_true"]
    eps_t_b3 = sel["B3"]["eps_true"]
    b1_budget = sel["B1"]["budget"]
    b2_budget = sel["B2"]["budget"]
    print("frozen config:", sel, flush=True)

    from mmdet.apis import init_detector
    from scqe.pf_runtime import PFRuntime
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    model.eval()
    rt = PFRuntime(model, device=args.device)
    enc = SupportEncoder(rt)
    supports = {}
    for seed in SEED_SUPPORT_PILOT:
        inst = support_instances(seed, 5)
        p, t, c_valid = enc.encode_episode(inst, n_classes=20)
        supports[seed] = dict(p=p, t=t, c_valid=c_valid)
        print(f"support {seed}: {int(c_valid.sum())}/20 valid", flush=True)

    entries = dcheck_entries()
    rows = []
    proposals = {}

    def save_prop(pk, qid, boxes, scores):
        proposals.setdefault(pk, {})[str(qid)] = dict(
            boxes=boxes.cpu().numpy().tolist(), scores=scores.cpu().numpy().tolist())

    def decode_plan(eng, plan):
        if plan is None or int((plan != eng.state.A0).sum()) == 0:
            return eng.state.b0_boxes, eng.state.b0_scores, 0.0
        y, s_pf, extras = rt.native_decode_R(eng.state, eng.R_of_plan(plan))
        tr = true_risk(y, eng.state.y0, eng.state.wsqrt)
        return (*rt.proposals_from_outputs(eng.state, extras[0], s_pf), tr)

    for qi, entry in enumerate(entries):
        qid = entry["image_id"]
        t0 = time.time()
        eng = QueryEngine(rt, "DIOR", qid).prepare(query_image_path(entry))
        t_prep = time.time() - t0
        for seed in SEED_SUPPORT_PILOT:
            base = dict(qid=qid, seed=seed, prep_s=t_prep)
            if eng.bypass_reason:
                rows.append(dict(**base, method="ALL",
                                 status=f"bypass_{eng.bypass_reason}"))
                continue
            sp = supports[seed]
            alpha, cols, _ = eng.build_alpha(sp["p"], sp["t"], sp["c_valid"])
            if alpha is None:
                rows.append(dict(**base, method="ALL", status="no_evidence"))
                continue
            save_prop(f"{seed}/B0", qid, eng.state.b0_boxes, eng.state.b0_scores)
            rows.append(dict(**base, method="B0", status="ok"))

            # B1
            out = b1_independent_rank(eng.ctx, alpha, cols, b1_budget)
            boxes, scores, tr = decode_plan(eng, out["plan"])
            save_prop(f"{seed}/B1", qid, boxes, scores)
            rows.append(dict(**base, method="B1", status=out["status"],
                             planned_m=out["m"], F0=out["F0"], Fplan=out["Fplan"],
                             true_risk=tr))
            # B2
            out = netgreedy(eng.ctx, alpha, cols, None, m_max=b2_budget)
            boxes, scores, tr = decode_plan(eng, out["plan"])
            save_prop(f"{seed}/B2", qid, boxes, scores)
            rows.append(dict(**base, method="B2", status=out["status"],
                             planned_m=out["m"], F0=out["F0"], Fplan=out["Fplan"],
                             true_risk=tr))
            # B3 (frozen eps_true)
            out = netgreedy(eng.ctx, alpha, cols, None, m_max=M_MAX)
            k = out["m"]
            best = None
            if k > 0:
                cum = eng.state.A0.clone()
                step_of = {i + 1: s for i, s in enumerate(out["steps"])}
                for m in sorted({min(v, k) for v in (1, 4, 8, 16, 32)} - {0}):
                    srec = step_of[m]
                    cum = cum.clone()
                    cum[srec["slot"]] = srec["incoming"]
                    boxes, scores, tr = decode_plan(eng, cum)
                    if tr <= eps_t_b3 and (best is None or m > best[0]):
                        best = (m, boxes, scores, tr)
            if best is None:
                save_prop(f"{seed}/B3", qid, eng.state.b0_boxes, eng.state.b0_scores)
                rows.append(dict(**base, method="B3", status="true_reject",
                                 planned_m=k, F0=out["F0"], Fplan=out["Fplan"]))
            else:
                save_prop(f"{seed}/B3", qid, best[1], best[2])
                rows.append(dict(**base, method="B3", status="accepted_prefix",
                                 planned_m=k, accepted_m=best[0],
                                 F0=out["F0"], Fplan=out["Fplan"], true_risk=best[3]))
            # M (frozen eps)
            out = netgreedy(eng.ctx, alpha, cols, eng.g, m_max=M_MAX,
                            eps_proxy=eps_p_m)
            if out["m"] == 0:
                save_prop(f"{seed}/M", qid, eng.state.b0_boxes, eng.state.b0_scores)
                rows.append(dict(**base, method="M", status="no_exchange",
                                 F0=out["F0"], Fplan=out["Fplan"]))
            else:
                boxes, scores, tr = decode_plan(eng, out["plan"])
                if tr <= eps_t_m:
                    save_prop(f"{seed}/M", qid, boxes, scores)
                    rows.append(dict(**base, method="M", status="accepted",
                                     planned_m=out["m"], accepted_m=out["m"],
                                     F0=out["F0"], Fplan=out["Fplan"], true_risk=tr))
                else:
                    save_prop(f"{seed}/M", qid, eng.state.b0_boxes, eng.state.b0_scores)
                    rows.append(dict(**base, method="M", status="true_reject",
                                     planned_m=out["m"], accepted_m=0,
                                     F0=out["F0"], Fplan=out["Fplan"], true_risk=tr))
        print(f"[{qi+1}/{len(entries)}] {qid} ({t_prep:.1f}s prep)", flush=True)

    with open(OUT / "per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for pk, val in proposals.items():
        json.dump(val, open(OUT / "proposals" / f"{pk.replace('/', '__')}.json", "w"))
    json.dump(dict(frozen_selections=sel), open(OUT / "frozen_config.json", "w"), indent=1)
    print("E3 stage-1 done")


if __name__ == "__main__":
    main()
