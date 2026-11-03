"""Classify audited, support-independent B0 proposals with another support state."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORKER = ROOT / "runs/scqe_v1/code/backend/classify_stream_worker.py"
BE_PY = Path("/home/root-user/.conda/envs/cdvito/bin/python")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    ap.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.audit.read_text())
    if not audit.get("formal_ready"):
        raise RuntimeError("formal B0 replay requires passing input/history audit")
    source = args.source.resolve()
    while not (source / "producer_done.json").exists():
        launch = json.loads((source / "launch.json").read_text())
        if launch["status"] == "FAILED":
            raise RuntimeError("source B0 producer failed")
        time.sleep(5)
    time.sleep(3)
    source_meta_path = source / "metadata.json"
    meta = json.loads(source_meta_path.read_text())
    count = meta.get("expected_queries")
    source_shot = meta.get("shot")
    if meta.get("protocol") != "D" or meta.get("dataset") != "DIOR" or \
            meta.get("methods") != ["B0"] or source_shot not in (1, 5, 10) or \
            count not in (40, 4900) or \
            (count == 40 and source_shot != args.shot):
        raise ValueError("source is not a frozen DIOR D B0 episode")
    if len(list((source / "jobs").glob("*.ready.json"))) != count or \
            len(list((source / "plans").glob("*__B0.json"))) != count:
        raise RuntimeError("source proposals/plans are incomplete")
    support = ROOT / f"runs/sqr_v1/splits/dior_support_{args.shot}shot_seed{args.seed}.json"
    backend = ROOT / f"runs/scqe_v1/backend/seed{args.seed}_shot{args.shot}"
    weights = backend / "adapt/model_final.pth"
    proto = backend / f"DIOR_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    for path in (support, weights, proto):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True)
    for name in ("jobs", "plans", "producer_done.json"):
        (args.out / name).symlink_to(source / name, target_is_directory=name != "producer_done.json")
    replay_meta = dict(meta, seed=args.seed, shot=args.shot,
                       support_sha256=sha(support),
                       backend_checkpoint_sha256=sha(weights),
                       backend_prototype_sha256=sha(proto),
                       gpu_backend=args.device,
                       shared_proposal_source=str(source),
                       shared_proposal_source_shot=source_shot,
                       shared_proposal_metadata_sha256=sha(source_meta_path),
                       reuse_rule="B0 native PF-RPN proposals are support/shot independent; "
                       "plans and query jobs are linked read-only to the source episode")
    atomic_json(args.out / "metadata.json", replay_meta)
    command = [str(BE_PY), str(WORKER), "--seed", str(args.seed),
               "--shot", str(args.shot), "--spool", str(args.out),
               "--device", args.device]
    launch = dict(status="RUNNING", start=time.time(), mode="B0_SHARED_PROPOSALS",
                  source=str(source), audit=str(args.audit), worker=command,
                  device=args.device)
    atomic_json(args.out / "launch.json", launch)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="1")
    with (args.out / "worker.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    launch.update(end=time.time(), producer_exit=0, worker_exit=process.returncode,
                  status="COMPLETE" if process.returncode == 0 else "FAILED")
    atomic_json(args.out / "launch.json", launch)
    if process.returncode:
        raise RuntimeError(f"B0 replay failed: {args.out / 'worker.log'}")


if __name__ == "__main__":
    main()
