"""Record each completed formal DIOR support while the next GPU episode runs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
RUN = ROOT / "runs/scqe_server_20261014"
SCRIPTS = ROOT / "runs/scqe_v1/code/scripts"
BE_PY = "/home/root-user/.conda/envs/cdvito/bin/python"
PF_PY = "/home/root-user/.conda/envs/sqr/bin/python"
SEEDS = (2023, 42, 1234, 7, 99)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=5)
    args = parser.parse_args()
    while True:
        complete = 0
        for seed in SEEDS:
            out = RUN / f"metrics/per_support_full/shot{args.shot}_seed{seed}.json"
            legacy_out = RUN / f"metrics/per_support_full/seed{seed}.json"
            if args.shot == 5 and legacy_out.exists() and not out.exists():
                out = legacy_out
            if out.exists():
                complete += 1
                continue
            source = RUN / f"runs/dior_D_{args.shot}_seed{seed}"
            if args.shot == 5 and seed == 1234 and \
                    (RUN / "runs/dior_D_5_seed1234_resume1").exists():
                source = RUN / "runs/dior_D_5_seed1234_resume1"
            if args.shot == 5 and seed == 1234 and \
                    (RUN / "runs/dior_D_5_seed1234_resume2").exists():
                source = RUN / "runs/dior_D_5_seed1234_resume2"
            launch = source / "launch.json"
            if not launch.exists():
                continue
            status = json.loads(launch.read_text()).get("status")
            if status == "FAILED":
                raise RuntimeError(f"formal GPU run failed: {source}")
            if status != "COMPLETE":
                continue
            subprocess.run([
                BE_PY, str(SCRIPTS / "evaluate_dior_support.py"),
                "--shot", str(args.shot),
                "--run", str(source),
                "--gt", str(RUN / "audit_attested/dior_final_gt.json"),
                "--out", str(out),
            ], cwd=ROOT, check=True)
            subprocess.run([PF_PY, str(SCRIPTS / "record_table1_status.py")],
                           cwd=ROOT, check=True)
            complete += 1
        if complete == len(SEEDS):
            print(f"all five formal {args.shot}-shot support episodes recorded", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"support monitor failed: {exc!r}", file=sys.stderr, flush=True)
        raise
