"""Save one complete 4,900-query DIOR support episode before all five finish."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=5)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    launch = json.loads((args.run / "launch.json").read_text())
    meta = json.loads((args.run / "metadata.json").read_text())
    worker = json.loads((args.run / "worker_done.json").read_text())
    producer = json.loads((args.run / "producer_done.json").read_text())
    if launch.get("status") != "COMPLETE" or meta.get("expected_queries") != 4900 or \
            meta.get("protocol") != "D" or meta.get("dataset") != "DIOR" or \
            meta.get("shot") != args.shot or meta.get("seed") not in (2023, 42, 1234, 7, 99):
        raise RuntimeError("support episode is not a complete held-out DIOR run")
    audit_path = Path(launch.get("audit") or "")
    if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("formal_ready"):
        raise RuntimeError("held-out audit is missing or not formal-ready")
    expected_split = Path(__file__).resolve().parents[2] / "splits/dior_final_queries.json"
    if meta.get("split_sha256") != digest(expected_split):
        raise RuntimeError("formal split hash mismatch")
    if worker.get("deterministic_algorithms") is not True or \
            worker.get("attention") != "DINOv2 standard GPU attention":
        raise RuntimeError("backend policy mismatch")
    if producer.get("no_risk_search_cache_verified", 0) < 3:
        raise RuntimeError("B2/B3 GPU search cache was not verified")
    assignment = (meta.get("gpu_proposal"), meta.get("gpu_backend"))
    mode = meta.get("execution_mode", "concurrent_two_gpu")
    if (assignment, mode) not in {
            (("cuda:0", "cuda:1"), "concurrent_two_gpu"),
            (("cuda:0", "cuda:0"), "sequential_single_gpu")} or \
            launch.get("execution_mode", "concurrent_two_gpu") != mode or \
            worker.get("device") != assignment[1]:
        raise RuntimeError("GPU assignment/provenance mismatch")
    methods = meta["methods"]
    if methods != ["B0", "B1", "B2", "B3", "SCQE"]:
        raise RuntimeError("wrong formal method set")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.gt))
    ids = set(gt.getImgIds())
    if len(ids) != 4900:
        raise RuntimeError("formal GT does not contain 4,900 query images")
    rows = []
    for method in methods:
        plans = list((args.run / "plans").glob(f"*__{method}.json"))
        if len(plans) != 4900:
            raise RuntimeError(f"missing {method} plans")
        prediction = args.run / f"predictions_{method}.json"
        predictions = json.loads(prediction.read_text())
        if not {int(det["image_id"]) for det in predictions}.issubset(ids):
            raise RuntimeError(f"{method} prediction IDs are outside formal GT")
        with contextlib.redirect_stdout(io.StringIO()):
            metric = evaluate(predictions, gt)
        rows.append(dict(run_id=args.run.name, protocol="D", dataset="DIOR",
                         shot=meta["shot"], support_seed=meta["seed"], method=method,
                         split_hash=meta["split_sha256"], support_hash=meta["support_sha256"],
                         config_hash=meta["lock_sha256"],
                         checkpoint_hash=meta["checkpoint_sha256"],
                         backend_checkpoint_hash=meta["backend_checkpoint_sha256"],
                         backend_prototype_hash=meta["backend_prototype_sha256"],
                         status="COMPLETE", **metric,
                         prediction_path=str(prediction), prediction_hash=digest(prediction)))
    result = dict(status="COMPLETE", run=str(args.run), gt_sha256=digest(args.gt),
                  worker_policy=worker, per_support=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, args.out)
    print(f"saved five method AP rows for support {meta['seed']}")


if __name__ == "__main__":
    main()
