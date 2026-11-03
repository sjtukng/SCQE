"""Run one DOTA-HBB D support episode sequentially on a selected GPU."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
BACKEND = SCRIPTS.parent / "backend"
SPLITS = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
PF_PY = Path("/home/root-user/.conda/envs/sqr/bin/python")
BE_PY = Path("/home/root-user/.conda/envs/cdvito/bin/python")
METHODS = ("B0", "B1", "B2", "B3", "SCQE")


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def run(command, log, env):
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                       check=True, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2023, 42, 1234, 7, 99), required=True)
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("DOTA formal output directories are write-once")
    for path in (args.lock, args.acceptance,
                 SPLITS / "data_audit.json", SPLITS / "integrity_report.json",
                 SPLITS / "val_chips_1024_gap200.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not json.loads(args.acceptance.read_text()).get("pass_all"):
        raise RuntimeError("SCQE E2 implementation acceptance has not passed")
    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "LOCKED" or lock.get("protocol") != "D" or \
            lock.get("dataset") != "DIOR":
        raise RuntimeError("DOTA must inherit the frozen DIOR D parameter lock")
    if json.loads((SPLITS / "integrity_report.json").read_text()).get("status") != "PASS":
        raise RuntimeError("DOTA chip integrity audit has not passed")
    backend = ROOT / f"runs/scqe_v1/backend_dota/seed{args.seed}_shot{args.shot}"
    for path in (backend / "adapt/model_final.pth",
                 backend / f"DOTA_HBB_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) <= args.physical_gpu:
        raise RuntimeError(f"CUDA GPU {args.physical_gpu} is unavailable")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.physical_gpu), OMP_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    args.out.mkdir(parents=True)
    launch = dict(status="RUNNING", start=time.time(), seed=args.seed, shot=args.shot,
                  methods=METHODS, dataset="DOTA-HBB", protocol="D",
                  execution_mode="sequential_single_gpu",
                  physical_gpu=args.physical_gpu,
                  gpu_lines=[gpu_lines[args.physical_gpu]],
                  lock=str(args.lock), acceptance=str(args.acceptance))
    write(args.out / "launch.json", launch)
    try:
        run([PF_PY, SCRIPTS / "infer_dota_stream.py", "--split",
             SPLITS / "val_chips_1024_gap200.json", "--seed", args.seed,
             "--shot", args.shot, "--methods", ",".join(METHODS),
             "--lock", args.lock, "--spool", args.out, "--record-state"],
            args.out / "producer.log", env)
        for method in METHODS:
            raw = args.out / f"raw_{method}.bin.gz"
            policy = args.out / f"raw_policy_{method}.json"
            run([BE_PY, BACKEND / "classify_dota_chip_raw.py",
                 "--seed", args.seed, "--shot", args.shot,
                 "--method", method, "--jobs", args.out / "jobs",
                 "--out", raw, "--policy", policy],
                args.out / f"classifier_{method}.log", env)
            run([BE_PY, SCRIPTS / "merge_dota_parent_detections.py",
                 "--chip-map", SPLITS / "val_chips_1024_gap200.json",
                 "--raw-jsonl", raw, "--raw-policy", policy,
                 "--out", args.out / f"predictions_{method}.json",
                 "--audit", args.out / f"parent_merge_{method}.json"],
                args.out / f"parent_merge_{method}.log", env)
        launch.update(status="COMPLETE", end=time.time())
        write(args.out / "launch.json", launch)
    except BaseException as exc:
        launch.update(status="FAILED", end=time.time(), error=repr(exc))
        write(args.out / "launch.json", launch)
        raise


if __name__ == "__main__":
    main()
