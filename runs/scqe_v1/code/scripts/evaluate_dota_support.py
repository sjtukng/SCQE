"""Evaluate one complete DOTA-HBB support on the 458 validation parents."""
import argparse
import hashlib
import json
from pathlib import Path

from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


ROOT = Path(__file__).resolve().parents[4]
SPLITS = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
METHODS = ("B0", "B1", "B2", "B3", "SCQE")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("DOTA support AP output is write-once")
    launch = json.loads((args.run / "launch.json").read_text())
    meta = json.loads((args.run / "metadata.json").read_text())
    done = json.loads((args.run / "producer_done.json").read_text())
    if launch.get("status") != "COMPLETE" or meta.get("dataset") != "DOTA-HBB" or \
            meta.get("protocol") != "D" or meta.get("expected_queries") != 5297 or \
            meta.get("methods") != list(METHODS) or \
            done.get("count") != 5297 or \
            done.get("no_risk_search_cache_verified", 0) < 3 or \
            meta.get("state_format") != "torch_save_gzip_level1" or \
            len(list((args.run / "states").glob("*.pt.gz"))) != 5297:
        raise RuntimeError("DOTA support episode or shared search verification is incomplete")
    if meta.get("split_sha256") != sha(SPLITS / "val_chips_1024_gap200.json") or \
            meta.get("support_sha256") != sha(SPLITS /
                f"dota_support_{meta['shot']}shot_seed{meta['seed']}.json"):
        raise RuntimeError("DOTA frozen query/support hashes differ")
    gt_path = SPLITS / "val_parents_hbb.json"
    gt = COCO(str(gt_path))
    ids = set(gt.getImgIds())
    if len(ids) != 458:
        raise RuntimeError("DOTA HBB GT must contain 458 validation parents")
    gt_counts = {}
    for ann in gt.dataset["annotations"]:
        gt_counts[ann["image_id"]] = gt_counts.get(ann["image_id"], 0) + 1
    gt_over_100 = sum(value > 100 for value in gt_counts.values())
    if gt_over_100 != 69:
        raise RuntimeError("DOTA validation GT>100 parent count differs from frozen audit")
    rows = []
    for method in METHODS:
        prediction = args.run / f"predictions_{method}.json"
        merge = json.loads((args.run / f"parent_merge_{method}.json").read_text())
        policy = json.loads((args.run / f"raw_policy_{method}.json").read_text())
        if merge.get("status") != "COMPLETE" or \
                merge.get("output_sha256") != sha(prediction) or \
                policy.get("checkpoint_sha256") != meta["backend_checkpoint_sha256"] or \
                policy.get("prototype_sha256") != meta["backend_prototype_sha256"]:
            raise RuntimeError(f"DOTA raw-to-parent provenance mismatch: {method}")
        predictions = json.loads(prediction.read_text())
        if any(int(row["image_id"]) not in ids for row in predictions):
            raise RuntimeError(f"DOTA prediction outside validation parents: {method}")
        metric = evaluate(predictions, gt)
        rows.append(dict(run_id=args.run.name, protocol="D", dataset="DOTA-HBB",
                         shot=meta["shot"], support_seed=meta["seed"], method=method,
                         split_hash=meta["split_sha256"],
                         support_hash=meta["support_sha256"],
                         config_hash=meta["lock_sha256"],
                         checkpoint_hash=meta["checkpoint_sha256"],
                         backend_checkpoint_hash=meta["backend_checkpoint_sha256"],
                         backend_prototype_hash=meta["backend_prototype_sha256"],
                         status="COMPLETE", **metric,
                         prediction_path=str(prediction), prediction_hash=sha(prediction),
                         parent_gt_over_100=gt_over_100,
                         parent_cap_affected=merge["counts"]["parents_affected_by_cap"],
                         detections_before_parent_cap=merge["counts"]["after_global_nms"],
                         detections_after_parent_cap=merge["counts"]["after_final_parent_cap"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(status="COMPLETE", run=str(args.run),
                                        gt_sha256=sha(gt_path), per_support=rows), indent=2))
    print(f"saved five DOTA parent-level AP rows for support {meta['seed']}")


if __name__ == "__main__":
    main()
