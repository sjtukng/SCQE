"""E2 stage-2: classify all E2 proposals, evaluate AP, select frozen config.

Selection rules (spec 6.3):
- M: cell with best mean paired AP (vs B0) across the 2 calibration support
  episodes; ties -> smaller eps_true, then smaller eps_proxy.
- B1/B2: budget in {8,16,32} with best mean paired AP (ties -> smaller budget).
- B3: independent true threshold selection on the same D_cal: for each
  candidate threshold, per query pick the highest-F prefix whose true risk
  <= threshold (replayed offline), rebuild combined detections, evaluate.
Writes calibration.csv, metrics json, frozen config (e2_frozen_config.json).
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
E2 = ROOT / "runs/scqe_v1/e2_calibration"
BACKEND = ROOT / "runs/scqe_v1/code/backend"
GT = ROOT / "runs/scqe_v1/splits/dcal40_gt.json"
SEEDS = (2023, 42)


def classify_and_eval(seed, run_key):
    """run_key like 'B0', 'B18', 'M_p1.243e-02_t4.756e-05'."""
    pf = E2 / "proposals" / f"{seed}__{run_key}.json"
    det = E2 / "detections" / f"{seed}__{run_key}.json"
    det.parent.mkdir(parents=True, exist_ok=True)
    if not det.exists():
        r = subprocess.run(
            [sys.executable, str(BACKEND / "classify_proposals.py"),
             "--seed", str(seed), "--shot", "5",
             "--proposals", str(pf), "--out", str(det),
             "--device", "cuda:1"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"classify failed {run_key}: {r.stderr[-400:]}")
    ev = subprocess.run(
        [sys.executable, "-c",
         "import sys,json; sys.path.insert(0,r'%s'); from offline_eval import evaluate; "
         "print(json.dumps(evaluate(json.load(open(sys.argv[1])), sys.argv[2])))"
         % BACKEND, str(det), str(GT)],
        capture_output=True, text=True)
    return json.loads(ev.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-classify", action="store_true")
    args = ap.parse_args()

    grid = json.load(open(E2 / "grid.json"))
    proxy_grid, true_grid = grid["proxy_grid"], grid["true_grid"]
    records = list(csv.DictReader(open(E2 / "calibration_records.csv")))

    metrics = {}
    runs_needed = set()
    for seed in SEEDS:
        runs_needed.add((seed, "B0"))
        for b in (8, 16, 32):
            runs_needed.add((seed, f"B1{b}"))
            runs_needed.add((seed, f"B2{b}"))
        for ep in proxy_grid:
            for et in true_grid:
                runs_needed.add((seed, f"M_p{ep:.3e}_t{et:.3e}"))
        for m in (1, 4, 8, 16, 32):
            if (E2 / "proposals" / f"{seed}__B3_prefix{m}.json").exists():
                runs_needed.add((seed, f"B3_prefix{m}"))
    # B3 prefix detections
    prefix_risks = {}
    for rec in records:
        if rec.get("method") == "B3" and rec.get("prefix_true_risk"):
            prefix_risks[(int(rec["qid"]), int(rec["seed"]))] = json.loads(
                rec["prefix_true_risk"])

    for seed, rk in sorted(runs_needed):
        if not (E2 / "proposals" / f"{seed}__{rk}.json").exists():
            print("missing proposals:", seed, rk)
            continue
        metrics[(seed, rk)] = classify_and_eval(seed, rk)
        print(f"{seed} {rk:32s} AP={metrics[(seed, rk)]['AP']:.2f}", flush=True)

    ap_b0 = {seed: metrics[(seed, "B0")]["AP"] for seed in SEEDS}

    rows = []
    # B1/B2 budgets
    sel = {}
    for method in ("B1", "B2"):
        best = None
        for b in (8, 16, 32):
            d = [metrics[(s, f"{method}{b}")]["AP"] - ap_b0[s] for s in SEEDS]
            md = sum(d) / len(d)
            rows.append(dict(method=method, budget=b, mean_paired_AP=md,
                             dets=str(dict(zip(SEEDS, d)))))
            if best is None or md > best[1] + 1e-9 or (abs(md - best[1]) <= 1e-9 and b < best[0]):
                best = (b, md)
        sel[method] = dict(budget=best[0], mean_paired_AP=best[1])
    # M cells
    best = None
    for ep in proxy_grid:
        for et in true_grid:
            rk = f"M_p{ep:.3e}_t{et:.3e}"
            d = [metrics[(s, rk)]["AP"] - ap_b0[s] for s in SEEDS]
            md = sum(d) / len(d)
            rows.append(dict(method="M", eps_proxy=ep, eps_true=et,
                             mean_paired_AP=md, dets=str(dict(zip(SEEDS, d)))))
            key = (-md, et, ep)
            if best is None or key < best[0]:
                best = (key, ep, et, md)
    sel["M"] = dict(eps_proxy=best[1], eps_true=best[2], mean_paired_AP=best[3])
    # B3 thresholds (replay prefixes)
    b3_eval = {}
    for seed in SEEDS:
        for et in true_grid:
            det_out = E2 / "detections" / f"{seed}__B3_thr{et:.3e}.json"
            if not det_out.exists():
                by_q = {}
                for m in (1, 4, 8, 16, 32):
                    src = E2 / "detections" / f"{seed}__B3_prefix{m}.json"
                    if not src.exists():
                        continue
                    per_q = {}
                    for d in json.load(open(src)):
                        per_q.setdefault(d["image_id"], []).append(d)
                    for qid, ds in per_q.items():
                        by_q.setdefault(qid, {})[m] = ds
                final = []
                b0d = json.load(open(E2 / "detections" / f"{seed}__B0.json"))
                for qid in set(list(by_q.keys()) + [x["image_id"] for x in b0d]):
                    pr = prefix_risks.get((qid, seed), {})
                    ok = [m for m in (1, 4, 8, 16, 32)
                          if m in by_q.get(qid, {}) and pr.get(str(m), 1e9) <= et]
                    if ok:
                        m = max(ok)
                        final.extend(by_q[qid][m])
                    else:
                        final.extend([x for x in b0d if x["image_id"] == qid])
                json.dump(final, open(det_out, "w"))
            ev = subprocess.run(
                [sys.executable, "-c",
                 "import sys,json; sys.path.insert(0,r'%s'); from offline_eval import evaluate; "
                 "print(json.dumps(evaluate(json.load(open(sys.argv[1])), sys.argv[2])))"
                 % BACKEND, str(det_out), str(GT)],
                capture_output=True, text=True)
            b3_eval[(seed, et)] = json.loads(ev.stdout.strip().splitlines()[-1])
    best = None
    for et in true_grid:
        d = [b3_eval[(s, et)]["AP"] - ap_b0[s] for s in SEEDS]
        md = sum(d) / len(d)
        rows.append(dict(method="B3", eps_true=et, mean_paired_AP=md,
                         dets=str(dict(zip(SEEDS, d)))))
        if best is None or md > best[1] + 1e-9 or (abs(md - best[1]) <= 1e-9 and et < best[0]):
            best = (et, md)
    sel["B3"] = dict(eps_true=best[0], mean_paired_AP=best[1])

    with open(E2 / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(dict(selections=sel, grid=grid,
                   ap_b0=ap_b0,
                   metrics={f"{s}__{k}": v for (s, k), v in metrics.items()}),
              open(E2 / "e2_frozen_config.json", "w"), indent=1)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
