"""Queue frozen DOTA-HBB protocol D after all single-GPU DIOR D work finishes.

Only the 15 support episodes enter CD-ViTO adaptation. Validation labels are
read by the separate evaluation commands after predictions are sealed.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
SCRIPTS = Path(__file__).resolve().parent
SPLITS = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
PF_PY = Path("/home/root-user/.conda/envs/sqr/bin/python")
BE_PY = Path("/home/root-user/.conda/envs/cdvito/bin/python")
SEEDS = (2023, 42, 1234, 7, 99)
PHYSICAL_GPU = int(os.environ.get("SCQE_PHYSICAL_GPU", "0"))
LOCK = RUN / "config_lock_D.json"
ACCEPT = RUN / "acceptance_fp64_vjp_complete/implementation_acceptance.json"
STATUS = RUN / "dota_D_scheduler_status.json"


def read(path):
    return json.loads(path.read_text())


def status(**fields):
    record = read(STATUS) if STATUS.is_file() else {}
    record.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, STATUS)


def run(command, log):
    print("running", " ".join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def wait_for_dior():
    targets = [RUN / f"metrics/dior_D_{shot}/metrics.json" for shot in (5, 10, 1)]
    status(state="WAITING_FOR_DIOR", completed_shots=[])
    while not all(path.is_file() for path in targets):
        for shot in (5, 10, 1):
            for seed in SEEDS:
                launch = RUN / f"runs/dior_D_{shot}_seed{seed}/launch.json"
                if launch.is_file() and read(launch).get("status") == "FAILED":
                    raise RuntimeError(f"DIOR episode failed before DOTA: {launch}")
        status(state="WAITING_FOR_DIOR",
               completed_shots=[shot for shot, path in zip((5, 10, 1), targets)
                                if path.is_file()])
        time.sleep(60)
    if any(len(read(path).get("per_support", [])) != 25 for path in targets):
        raise RuntimeError("DIOR D five-support metrics are incomplete")
    if read(LOCK).get("status") != "LOCKED" or not read(ACCEPT).get("pass_all"):
        raise RuntimeError("DOTA cannot inherit the DIOR D lock or E2 acceptance")


def require_space():
    free = shutil.disk_usage(ROOT).free
    if free < 100 * (1 << 30):
        raise RuntimeError(f"less than 100 GiB free before DOTA episode: {free}")


def one_support(shot, seed):
    require_space()
    status(state="ADAPTING", shot=shot, seed=seed)
    backend = ROOT / f"runs/scqe_v1/backend_dota/seed{seed}_shot{shot}"
    checkpoint = backend / "adapt/model_final.pth"
    prototype = backend / f"DOTA_HBB_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl"
    if not checkpoint.is_file() or not prototype.is_file():
        run(["bash", ROOT / "runs/scqe_v1/code/run_dota_adaptation.sh", seed, shot],
            RUN / f"adaptation_logs/dota_seed{seed}_shot{shot}.log")
    if not checkpoint.is_file() or not prototype.is_file():
        raise RuntimeError(f"DOTA support adaptation incomplete: {backend}")
    output = RUN / f"runs/dota_D_{shot}_seed{seed}"
    status(state="INFERENCING", shot=shot, seed=seed)
    if not output.exists():
        run([PF_PY, SCRIPTS / "run_dota_pair.py", "--seed", seed,
             "--shot", shot, "--lock", LOCK, "--acceptance", ACCEPT,
             "--out", output, "--physical-gpu", PHYSICAL_GPU],
            RUN / f"dota_D_{shot}_launcher.log")
    if read(output / "launch.json").get("status") != "COMPLETE":
        raise RuntimeError(f"DOTA immutable episode incomplete: {output}")
    single = RUN / f"metrics/per_support_dota/shot{shot}_seed{seed}.json"
    status(state="EVALUATING", shot=shot, seed=seed)
    if not single.exists():
        run([BE_PY, SCRIPTS / "evaluate_dota_support.py", "--run", output,
             "--out", single], RUN / f"metrics/dota_D_{shot}_support_eval.log")
    if read(single).get("status") != "COMPLETE" or \
            len(read(single).get("per_support", [])) != 5:
        raise RuntimeError(f"DOTA per-support AP incomplete: {single}")
    run([PF_PY, SCRIPTS / "record_table1_status.py"],
        RUN / f"metrics/dota_D_{shot}_status.log")


def aggregate(shot, bootstrap):
    name = f"dota_D_{shot}" + ("" if bootstrap else "_primary")
    output = RUN / "metrics" / name
    if not output.exists():
        command = [BE_PY, SCRIPTS / "evaluate_runs.py",
                   "--protocol", "D", "--dataset", "DOTA-HBB", "--shot", shot,
                   "--gt", SPLITS / "val_parents_hbb.json"]
        for seed in SEEDS:
            command.extend(["--run", f"{seed}:{RUN / f'runs/dota_D_{shot}_seed{seed}'}"])
        if bootstrap:
            command.append("--bootstrap")
        command.extend(["--out", output])
        run(command, RUN / f"metrics/{name}.log")
    metrics = read(output / "metrics.json")
    if len(metrics.get("per_support", [])) != 25:
        raise RuntimeError(f"DOTA aggregate AP incomplete: {output}")
    run([PF_PY, SCRIPTS / "record_table1_status.py"],
        RUN / f"metrics/dota_D_{shot}_status.log")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    wait_for_dior()
    for shot in (5, 10, 1):
        for seed in SEEDS:
            one_support(shot, seed)
        status(state="AGGREGATING_PRIMARY", shot=shot, seed=None)
        aggregate(shot, bootstrap=False)
    for shot in (5, 10, 1):
        status(state="BOOTSTRAPPING", shot=shot, seed=None)
        aggregate(shot, bootstrap=True)
    run([PF_PY, SCRIPTS / "build_report.py", "--run-root", RUN],
        RUN / "metrics/dota_D_status.log")
    status(state="COMPLETE", shot=None, seed=None)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        status(state="FAILED", error=repr(exc))
        print(f"DOTA D scheduler failed: {exc!r}", file=sys.stderr, flush=True)
        raise
