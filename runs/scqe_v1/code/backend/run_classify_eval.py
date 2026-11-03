"""Batch: classify all proposals jsons in a directory and evaluate AP.

For each proposals_*.json -> detections_*.json via classify_proposals.py
(cdvito env), then COCO AP vs a GT json (offline).

Usage:
  python run_classify_eval.py --proposals-dir <dir> --gt <gt.json> \
      --seed 2023 --shot 5 [--pattern "B0*"] [--skip-classify]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
BACKEND = ROOT / "runs/scqe_v1/code/backend"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    pdir = Path(args.proposals_dir)
    out_rows = []
    for pf in sorted(pdir.glob(f"proposals_{args.pattern}.json")):
        det = pf.with_name(pf.name.replace("proposals_", "detections_"))
        if not args.skip_classify or not det.exists():
            cmd = [sys.executable, str(BACKEND / "classify_proposals.py"),
                   "--seed", str(args.seed), "--shot", str(args.shot),
                   "--proposals", str(pf), "--out", str(det),
                   "--device", args.device]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("CLASSIFY FAIL", pf.name, r.stderr[-500:])
                continue
            print(r.stdout.strip().splitlines()[-1])
        ev = subprocess.run(
            [sys.executable, "-c",
             "import sys,json; sys.path.insert(0,'%s'); from offline_eval import evaluate; "
             "print(json.dumps(evaluate(json.load(open(sys.argv[1])), sys.argv[2])))"
             % BACKEND, str(det), args.gt],
            capture_output=True, text=True)
        if ev.returncode != 0:
            print("EVAL FAIL", det.name, ev.stderr[-300:])
            continue
        m = json.loads(ev.stdout.strip().splitlines()[-1])
        m["run"] = pf.name.replace("proposals_", "").replace(".json", "")
        out_rows.append(m)
        print(f"{m['run']:40s} AP={m['AP'] if m['AP'] is not None else float('nan'):.2f}")
    out = pdir / f"metrics_seed{args.seed}.json"
    json.dump(out_rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
