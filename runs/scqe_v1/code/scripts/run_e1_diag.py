"""E1 diagnostics (first-batch prompt item 8).

Phase 1: on the first 12 D_cal images x 5-shot x supports {2023,42}, run
  B0, B1-Diagnostic32, NetGreedy-Diagnostic32 (B2); save all non-empty plans.
Phase 2: offline, per saved plan compute proxy risk (cached VJP) and true risk
  (one native decoder forward; repeat 3x for noise) -> s_p, s_t scales.
Phase 3: M diagnostic with (s_p, s_t), B3 diagnostic with min(s_t,1); record
  behavior (planned/accepted/bypass/reject), costs, peak memory; save proposals.

Writes e1_validation/diag_plans.json, scales.json, per_query.csv,
proposals_{method}_seed{S}.json under e1_validation/.
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

from scqe.config import (CHECKPOINT_PATH, CONFIG_PATH, M_MAX, P_PROBES,
                         SEED_GLOBAL, SEED_SUPPORT_CAL)
from scqe.data import dcal_entries, query_image_path, support_instances
from scqe.pipeline import QueryEngine
from scqe.risk import proxy_from_zeta, true_risk
from scqe.solver import netgreedy, b1_independent_rank
from scqe.support import SupportEncoder

OUT = Path(__file__).resolve().parents[2] / "e1_validation"
OUT.mkdir(parents=True, exist_ok=True)


def plan_proxy(g, zeta_init, steps, xi_vs, R0):
    """Replay zeta for a plan's steps (from A0)."""
    from scqe.risk import step_zeta
    zeta = zeta_init.clone()
    for s in steps:
        r = s["slot"]
        j_pos = s["j_pos"]
        zeta = step_zeta(g, zeta, r, xi_vs[j_pos], R0)
    return proxy_from_zeta(zeta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--phase", default="all", choices=["all", "1", "2", "3"])
    args = ap.parse_args()
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    model.eval()
    entries = dcal_entries()[: args.n]

    if args.phase in ("all", "1"):
        from scqe.pf_runtime import PFRuntime
        rt = PFRuntime(model, device=args.device)
        enc = SupportEncoder(rt)
        supports = {}
        for seed in SEED_SUPPORT_CAL:
            inst = support_instances(seed, 5)
            t0 = time.time()
            p, t, c_valid = enc.encode_episode(inst, n_classes=20)
            supports[seed] = dict(p=p, t=t, c_valid=c_valid)
            print(f"support seed {seed}: {int(c_valid.sum())}/20 classes valid, "
                  f"{time.time()-t0:.1f}s", flush=True)

        engines, alphas = {}, {}
        plans = []
        proposals_out = {}
        for qi, entry in enumerate(entries):
            qid = entry["image_id"]
            eng = QueryEngine(rt, "DIOR", qid).prepare(query_image_path(entry))
            engines[qid] = eng
            if eng.bypass_reason:
                print(f"[{qi+1}/{len(entries)}] {qid} bypass={eng.bypass_reason}")
                continue
            for seed in SEED_SUPPORT_CAL:
                sp = supports[seed]
                alpha, cols, st = eng.build_alpha(sp["p"], sp["t"], sp["c_valid"])
                if alpha is None:
                    continue
                alphas[(qid, seed)] = (alpha, cols)
                for method in ("B0", "B1", "B2"):
                    t0 = time.time()
                    if method == "B0":
                        out = dict(m=0, status="ok", F0=None, Fplan=None, steps=[])
                        boxes, scores = eng.state.b0_boxes, eng.state.b0_scores
                    else:
                        if method == "B1":
                            out = b1_independent_rank(eng.ctx, alpha, cols, M_MAX)
                        else:
                            out = netgreedy(eng.ctx, alpha, cols, None,
                                            m_max=M_MAX)
                        if out["m"] > 0:
                            y, s_pf, extras = rt.native_decode_R(
                                eng.state, eng.R_of_plan(out["plan"]))
                            boxes, scores = rt.proposals_from_outputs(
                                eng.state, extras[0], s_pf)
                        else:
                            boxes, scores = eng.state.b0_boxes, eng.state.b0_scores
                    pk = f"{method}_seed{seed}"
                    proposals_out.setdefault(pk, {})[str(qid)] = dict(
                        boxes=boxes.cpu().numpy().tolist(),
                        scores=scores.cpu().numpy().tolist())
                    if method == "B0":
                        continue
                    rec = dict(qid=qid, seed=seed, method=method,
                               m=out["m"], status=out["status"],
                               F0=out["F0"], Fplan=out["Fplan"],
                               time=time.time() - t0,
                               steps=[dict(slot=s["slot"], out=s["out"],
                                           incoming=s["incoming"],
                                           gain=s["gain"], j_pos=int(
                                               torch.nonzero(eng.ctx.vs_idx == s["incoming"]).squeeze()))
                                      for s in out["steps"]])
                    plans.append(rec)
            print(f"[{qi+1}/{len(entries)}] {qid} done", flush=True)
        json.dump(plans, open(OUT / "diag_plans.json", "w"), indent=1)
        for pk, val in proposals_out.items():
            json.dump(val, open(OUT / f"proposals_{pk}.json", "w"))
        print(f"phase1 done: {len(plans)} non-bypass method plans")

    if args.phase in ("all", "2", "3"):
        plans = json.load(open(OUT / "diag_plans.json"))
        if args.phase in ("all", "2"):
            print("NOTE: phases 2/3 re-run the model; run phase 1 separately "
                  "first if diag_plans.json is stale.")
        from scqe.pf_runtime import PFRuntime
        rt = PFRuntime(model, device=args.device)
        enc = SupportEncoder(rt)
        supports = {}
        for seed in SEED_SUPPORT_CAL:
            inst = support_instances(seed, 5)
            p, t, c_valid = enc.encode_episode(inst, n_classes=20)
            supports[seed] = dict(p=p, t=t, c_valid=c_valid)

        # phase 2: scales
        scales = {}
        if args.phase in ("all", "2"):
            proxy_vals, true_vals = [], []
            noise_true = []
            engines = {}
            done = set()
            for rec in plans:
                key = rec["qid"]
                if key not in engines:
                    e = next(en for en in entries if en["image_id"] == key)
                    engines[key] = QueryEngine(rt, "DIOR", key).prepare(
                        query_image_path(e))
                eng = engines[key]
                if eng.bypass_reason or rec["m"] == 0:
                    continue
                sp = supports[rec["seed"]]
                alpha, cols, _ = eng.build_alpha(sp["p"], sp["t"], sp["c_valid"])
                if alpha is None:
                    continue
                # proxy from plan
                plan = eng.state.A0.clone()
                for s in rec["steps"]:
                    plan[s["slot"]] = s["incoming"]
                dR = (eng.state.xi_all[plan] - eng.state.R0).to(torch.float64)
                direct = torch.einsum("prc,rc->p", eng.g.to(torch.float64), dR)
                pr = proxy_from_zeta(direct)
                if pr > 1e-12:
                    proxy_vals.append(pr)
                # true risk: 3 repeats for noise
                trs = []
                for _ in range(3):
                    y, _, _ = rt.native_decode_R(eng.state, eng.R_of_plan(plan))
                    trs.append(true_risk(y, eng.state.y0, eng.state.wsqrt))
                noise_true.append(max(trs) - min(trs))
                tr = trs[0]
                if tr > 1e-12:
                    true_vals.append(tr)
            nu_t = max(1e-12, 10.0 * (max(noise_true) if noise_true else 0.0))
            nu_p = 1e-12  # proxy is analytic from cached g; no forward noise
            s_p = float(np.median([v for v in proxy_vals if v > nu_p])) if proxy_vals else None
            s_t = float(np.median([v for v in true_vals if v > nu_t])) if true_vals else None
            scales = dict(s_p=s_p, s_t=s_t, nu_p=nu_p, nu_t=nu_t,
                          n_plans=len(plans),
                          n_proxy_pos=len(proxy_vals), n_true_pos=len(true_vals),
                          proxy_vals=proxy_vals, true_vals=true_vals,
                          noise_true_max=max(noise_true) if noise_true else None)
            json.dump(scales, open(OUT / "scales.json", "w"), indent=1)
            print("scales:", {k: scales[k] for k in ("s_p", "s_t", "nu_p", "nu_t",
                                                     "n_proxy_pos", "n_true_pos")})
        else:
            scales = json.load(open(OUT / "scales.json"))

        if args.phase in ("all", "3") and scales.get("s_p") and scales.get("s_t"):
            eps_p, eps_t = scales["s_p"], scales["s_t"]
            rows = []
            proposals_out = {}
            engines = {}
            for qi, entry in enumerate(entries):
                qid = entry["image_id"]
                if qid not in engines:
                    engines[qid] = QueryEngine(rt, "DIOR", qid).prepare(
                        query_image_path(entry))
                eng = engines[qid]
                for seed in SEED_SUPPORT_CAL:
                    key = (qid, seed)
                    if eng.bypass_reason:
                        rows.append(dict(qid=qid, seed=seed, method="M",
                                         status=f"bypass_{eng.bypass_reason}"))
                        continue
                    sp = supports[seed]
                    alpha, cols, _ = eng.build_alpha(sp["p"], sp["t"], sp["c_valid"])
                    t0 = time.time()
                    mem0 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    res = eng.run_method("M", alpha, cols, M_MAX,
                                         eps_proxy=eps_p, eps_true=eps_t)
                    res3 = eng.run_method("B3", alpha, cols, M_MAX,
                                          eps_true=min(scales["s_t"], 1.0))
                    mem1 = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    for method, rr in (("M", res), ("B3", res3)):
                        rows.append(dict(
                            qid=qid, seed=seed, method=method,
                            planned_m=rr["planned_m"], accepted_m=rr["accepted_m"],
                            status=rr["status"], F0=rr["F0"], Fplan=rr["Fplan"],
                            true_risk=rr.get("true_risk"),
                            time=time.time() - t0, peak_mem_mb=(mem1 - mem0) / 1e6))
                        pk = f"{method}_seed{seed}"
                        proposals_out.setdefault(pk, {})[str(qid)] = dict(
                            boxes=rr["boxes"].cpu().numpy().tolist(),
                            scores=rr["scores"].cpu().numpy().tolist())
                print(f"[{qi+1}/{len(entries)}] {qid} M/B3 diag done", flush=True)
            with open(OUT / "per_query_diag.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            for pk, val in proposals_out.items():
                json.dump(val, open(OUT / f"proposals_{pk}.json", "w"))
            print("phase3 done")


if __name__ == "__main__":
    main()
