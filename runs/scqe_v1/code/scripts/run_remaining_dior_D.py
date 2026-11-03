"""After the primary 5-shot table, finish the available DIOR D 10/1-shot cells."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
SCRIPTS = Path(__file__).resolve().parent
SPLITS = ROOT / "runs/scqe_v1/splits"
PF_PY = "/home/root-user/.conda/envs/sqr/bin/python"
BE_PY = "/home/root-user/.conda/envs/cdvito/bin/python"
SEEDS = (2023, 42, 1234, 7, 99)
PHYSICAL_GPU = int(os.environ.get("SCQE_PHYSICAL_GPU", "0"))
LOCK = RUN / "config_lock_D.json"
ACCEPT = RUN / "acceptance_fp64_vjp_complete/implementation_acceptance.json"
AUDIT = RUN / "audit_attested/data_audit.json"
GT = RUN / "audit_attested/dior_final_gt.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    print("running", " ".join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def wait_for_five_shot():
    target = RUN / "metrics/dior_D_5_primary/metrics.json"
    while not target.is_file():
        for seed in (2023, 42, 1234):
            launch = RUN / f"confirmation/seed{seed}/launch.json"
            if launch.is_file() and read(launch).get("status") == "FAILED":
                raise RuntimeError(f"5-shot confirmation failed: {launch}")
        for seed in SEEDS:
            launch = RUN / f"runs/dior_D_5_seed{seed}/launch.json"
            if launch.is_file() and read(launch).get("status") == "FAILED":
                raise RuntimeError(f"5-shot formal episode failed: {launch}")
        time.sleep(60)
    metrics = read(target)
    if len(metrics.get("per_support", [])) != 25:
        raise RuntimeError("five-shot primary metrics are incomplete")
    print("five-shot primary AP complete; starting remaining DIOR D shots", flush=True)


def ensure_adaptation(shot, seed):
    backend = ROOT / f"runs/scqe_v1/backend/seed{seed}_shot{shot}"
    checkpoint = backend / "adapt/model_final.pth"
    prototype = backend / f"DIOR_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl"
    if not checkpoint.is_file() or not prototype.is_file():
        if shot != 1:
            raise FileNotFoundError(f"missing frozen 10-shot backend: {backend}")
        run(["env", f"CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}", "bash",
             ROOT / "runs/scqe_v1/code/run_adaptation.sh", seed, shot],
            RUN / f"adaptation_logs/remaining_seed{seed}_shot{shot}.log")
    if not checkpoint.is_file() or not prototype.is_file():
        raise FileNotFoundError(f"adaptation incomplete: {backend}")


def common_args(shot):
    args = ["--protocol", "D", "--dataset", "DIOR", "--shot", shot, "--gt", GT]
    for seed in SEEDS:
        args += ["--run", f"{seed}:{RUN / f'runs/dior_D_{shot}_seed{seed}'}"]
    return args


def run_shot(shot):
    if not read(AUDIT).get("formal_ready") or not read(ACCEPT).get("pass_all") or \
            read(LOCK).get("status") != "LOCKED":
        raise RuntimeError("formal audit, E2 or D lock is not ready")
    lock_hash = digest(LOCK)
    for seed in SEEDS:
        ensure_adaptation(shot, seed)
        out = RUN / f"runs/dior_D_{shot}_seed{seed}"
        if not out.exists():
            run([PF_PY, SCRIPTS / "run_dior_pair.py", "--split",
                 SPLITS / "dior_final_queries.json", "--seed", seed,
                 "--shot", shot, "--methods", "B0,B1,B2,B3,SCQE",
                 "--lock", LOCK, "--acceptance", ACCEPT, "--audit", AUDIT,
                 "--out", out, "--record-state", "--same-job-cache", "--single-gpu",
                 "--physical-gpu", PHYSICAL_GPU],
                RUN / f"dior_D_{shot}_launcher.log")
        launch = read(out / "launch.json")
        meta = read(out / "metadata.json")
        if launch.get("status") != "COMPLETE" or meta.get("lock_sha256") != lock_hash or \
                meta.get("shot") != shot or meta.get("seed") != seed:
            raise RuntimeError(f"incomplete or mismatched immutable run: {out}")
        single = RUN / f"metrics/per_support_full/shot{shot}_seed{seed}.json"
        if not single.exists():
            run([BE_PY, SCRIPTS / "evaluate_dior_support.py", "--run", out,
                 "--shot", shot, "--gt", GT, "--out", single],
                RUN / f"metrics/dior_D_{shot}_support_eval.log")
        run([PF_PY, SCRIPTS / "record_table1_status.py"],
            RUN / f"metrics/dior_D_{shot}_status.log")
    primary = RUN / f"metrics/dior_D_{shot}_primary"
    if not primary.exists():
        run([BE_PY, SCRIPTS / "evaluate_runs.py", *common_args(shot), "--out", primary],
            RUN / f"metrics/dior_D_{shot}_primary.log")
    if not (primary / "metrics.json").is_file():
        raise RuntimeError(f"primary metrics incomplete: {primary}")
    run([PF_PY, SCRIPTS / "record_table1_status.py"],
        RUN / f"metrics/dior_D_{shot}_status.log")


def bootstrap_shot(shot):
    bootstrap = RUN / f"metrics/dior_D_{shot}"
    if not bootstrap.exists():
        run([BE_PY, SCRIPTS / "evaluate_runs.py", *common_args(shot),
             "--bootstrap", "--out", bootstrap],
            RUN / f"metrics/dior_D_{shot}_bootstrap.log")
    if not (bootstrap / "metrics.json").is_file():
        raise RuntimeError(f"bootstrap metrics incomplete: {bootstrap}")
    run([PF_PY, SCRIPTS / "record_table1_status.py"],
        RUN / f"metrics/dior_D_{shot}_status.log")
    run([PF_PY, SCRIPTS / "build_report.py", "--run-root", RUN],
        RUN / f"metrics/dior_D_{shot}_status.log")
    print(f"DIOR D {shot}-shot complete", flush=True)


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    wait_for_five_shot()
    for shot in (10, 1):
        run_shot(shot)
        if shot == 1:
            run([PF_PY, SCRIPTS / "summarize_adaptation.py"],
                RUN / "adaptation_logs/remaining_summary.log")
    for shot in (10, 1):
        bootstrap_shot(shot)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"remaining DIOR D scheduler failed: {exc!r}", file=sys.stderr, flush=True)
        raise
