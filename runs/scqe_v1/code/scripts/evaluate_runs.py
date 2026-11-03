"""Global COCO AP and shared-parent paired bootstrap from five frozen supports."""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resampled_coco(original, image_ids):
    cats = original.dataset["categories"]
    image_lookup = {int(x["id"]): x for x in original.dataset["images"]}
    ann_lookup = {}
    for ann in original.dataset["annotations"]:
        ann_lookup.setdefault(int(ann["image_id"]), []).append(ann)
    images, anns = [], []
    aid = 1
    for new_id, old_id in enumerate(image_ids, 1):
        image = dict(image_lookup[int(old_id)], id=new_id)
        images.append(image)
        for original_ann in ann_lookup.get(int(old_id), []):
            anns.append(dict(original_ann, id=aid, image_id=new_id))
            aid += 1
    coco = COCO()
    coco.dataset = dict(info=original.dataset.get("info", {}),
                        licenses=original.dataset.get("licenses", []),
                        images=images, annotations=anns, categories=cats)
    coco.createIndex()
    return coco


def resampled_predictions(predictions, draws):
    by_parent = {}
    for det in predictions:
        by_parent.setdefault(int(det["image_id"]), []).append(det)
    result = []
    for new_id, old_id in enumerate(draws, 1):
        result.extend(dict(det, image_id=new_id) for det in by_parent.get(int(old_id), []))
    return result


def cached_bootstrap_ap(gt, predictions, positions):
    """Reuse frozen per-parent COCO matches for exact parent resampling.

    A resampled parent is an identical copy with a new image ID. COCO's
    per-image matching therefore stays identical; only the global ranking and
    GT count in accumulate() change. One area and maxDet=100 reproduce AP.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            detected = gt.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = dict(images=gt.dataset["images"],
                                    categories=gt.dataset["categories"], annotations=[])
            detected.createIndex()
        original = COCOeval(gt, detected, "bbox")
        original.params.maxDets = [100]
        original.params.areaRng = [[0, 1e10]]
        original.params.areaRngLbl = ["all"]
        original.evaluate()
    n_images = len(original._paramsEval.imgIds)
    if len(original.evalImgs) % n_images:
        raise RuntimeError("COCO parent match cache has inconsistent dimensions")
    blocks = [original.evalImgs[i:i + n_images]
              for i in range(0, len(original.evalImgs), n_images)]
    ap = np.empty(len(positions), dtype=np.float64)
    with contextlib.redirect_stdout(io.StringIO()):
        for iteration, draw in enumerate(positions):
            if len(draw) != n_images:
                raise RuntimeError("bootstrap draw must preserve parent count")
            sampled = copy.copy(original)
            sampled.params = copy.deepcopy(original.params)
            sampled._paramsEval = copy.deepcopy(original._paramsEval)
            new_ids = list(range(1, n_images + 1))
            sampled.params.imgIds = new_ids
            sampled._paramsEval.imgIds = new_ids
            sampled.evalImgs = [block[int(i)] for block in blocks for i in draw]
            sampled.accumulate()
            precision = sampled.eval["precision"][:, :, :, 0, 0]
            valid = precision[precision >= 0]
            ap[iteration] = float(valid.mean() * 100) if valid.size else float("nan")
    if not np.isfinite(ap).all():
        raise RuntimeError("bootstrap AP undefined for a frozen method/support")
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="seed:path; exactly five completed episode directories")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--protocol", choices=("D", "K"), required=True)
    ap.add_argument("--dataset", choices=("DIOR", "DOTA-HBB"), required=True)
    ap.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    runs = {}
    for item in args.run:
        seed, path = item.split(":", 1)
        runs[int(seed)] = Path(path)
    if len(runs) != 5:
        raise ValueError("formal comparison requires five distinct support seeds")
    gt = COCO(str(args.gt))
    ids = sorted(gt.getImgIds())
    predictions = {}
    rows = []
    split_hashes, checkpoint_hashes, config_hashes = set(), set(), set()
    methods = None
    for seed, path in sorted(runs.items()):
        launch = json.loads((path / "launch.json").read_text())
        meta = json.loads((path / "metadata.json").read_text())
        if launch["status"] != "COMPLETE":
            raise RuntimeError(f"incomplete run: {path}")
        if args.dataset == "DOTA-HBB":
            if launch.get("execution_mode") != "sequential_single_gpu" or \
                    meta.get("execution_mode") != "sequential_single_gpu":
                raise RuntimeError(f"DOTA single-GPU policy mismatch: {path}")
            done = json.loads((path / "producer_done.json").read_text())
            if done.get("status") != "COMPLETE" or done.get("count") != 5297 or \
                    done.get("no_risk_search_cache_verified", 0) < 3 or \
                    meta.get("state_format") != "torch_save_gzip_level1" or \
                    len(list((path / "states").glob("*.pt.gz"))) != 5297:
                raise RuntimeError(f"DOTA proposal verification incomplete: {path}")
        else:
            worker = json.loads((path / "worker_done.json").read_text())
            if worker.get("deterministic_algorithms") is not True or \
                    worker.get("attention") != "DINOv2 standard GPU attention":
                raise RuntimeError(f"formal backend policy mismatch: {path}")
        if meta["shot"] != args.shot or meta["seed"] != seed or \
                meta["dataset"] != args.dataset or meta["protocol"] != args.protocol:
            raise ValueError(f"run metadata mismatch: {path}")
        if meta.get("probes") != 16 or meta.get("budget_override") is not None:
            raise ValueError("formal AP table requires locked P=16, M=32")
        if methods is None:
            methods = meta["methods"]
        elif set(methods) != set(meta["methods"]):
            raise ValueError("methods differ between support episodes")
        split_hashes.add(meta["split_sha256"])
        checkpoint_hashes.add(meta["checkpoint_sha256"])
        config_hashes.add(meta["lock_sha256"])
        expected_queries = 5297 if args.dataset == "DOTA-HBB" else len(ids)
        if meta["expected_queries"] != expected_queries:
            raise ValueError(f"query/chip count differs from frozen dataset: {path}")
        for method in methods:
            if len(list((path / "plans").glob(f"*__{method}.json"))) != expected_queries:
                raise RuntimeError(f"missing query plans for {path}/{method}")
            pred_path = path / f"predictions_{method}.json"
            if args.dataset == "DOTA-HBB":
                merge = json.loads((path / f"parent_merge_{method}.json").read_text())
                policy = json.loads((path / f"raw_policy_{method}.json").read_text())
                if merge.get("status") != "COMPLETE" or \
                        merge.get("output_sha256") != digest(pred_path) or \
                        policy.get("checkpoint_sha256") != meta.get("backend_checkpoint_sha256") or \
                        policy.get("prototype_sha256") != meta.get("backend_prototype_sha256"):
                    raise RuntimeError(f"DOTA parent-level provenance mismatch: {path}/{method}")
            preds = json.loads(pred_path.read_text())
            if not {int(p["image_id"]) for p in preds}.issubset(set(ids)):
                raise ValueError("predictions contain IDs outside evaluation GT")
            predictions[(seed, method)] = preds
            metric = evaluate(preds, gt)
            rows.append(dict(run_id=path.name, protocol=args.protocol, dataset=args.dataset,
                             shot=args.shot, support_seed=seed, method=method,
                             split_hash=meta["split_sha256"], support_hash=meta["support_sha256"],
                             config_hash=meta["lock_sha256"],
                             checkpoint_hash=meta["checkpoint_sha256"],
                             backend_checkpoint_hash=meta.get("backend_checkpoint_sha256"),
                             backend_prototype_hash=meta.get("backend_prototype_sha256"),
                             status="COMPLETE", AP=metric["AP"], AP50=metric["AP50"],
                             AP75=metric["AP75"], AP_small=metric["AP_small"],
                             AP_medium=metric["AP_medium"], AP_large=metric["AP_large"],
                             per_class_AP=metric["per_class_AP"],
                             prediction_path=str(pred_path), prediction_hash=digest(pred_path)))
    if any(len(group) != 1 for group in (split_hashes, checkpoint_hashes, config_hashes)):
        raise ValueError("five episodes differ in split, checkpoint or config")
    summary = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        values = [row["AP"] for row in method_rows]
        if len(values) != 5 or any(x is None for x in values):
            raise RuntimeError(f"AP undefined for {method}")
        summary[method] = dict(raw_support_seeds=[row["support_seed"] for row in method_rows],
                               raw_support_AP=values, mean_AP=statistics.mean(values),
                               sample_SD=statistics.stdev(values))
    paired = {}
    if "SCQE" in methods:
        for baseline in ("B0", "B3"):
            if baseline in methods:
                paired[baseline] = statistics.mean(
                    next(row["AP"] for row in rows if row["support_seed"] == seed and row["method"] == "SCQE") -
                    next(row["AP"] for row in rows if row["support_seed"] == seed and row["method"] == baseline)
                    for seed in runs)
    bootstrap = {}
    if args.bootstrap and paired:
        rng = np.random.default_rng(20261013)
        parent_ids = np.asarray(ids, dtype=np.int64)
        positions = [np.searchsorted(parent_ids,
                                     rng.choice(parent_ids, size=len(parent_ids), replace=True))
                     for _ in range(2000)]
        scores = {}
        for seed in runs:
            for method in ("SCQE", *paired):
                scores[(seed, method)] = cached_bootstrap_ap(
                    gt, predictions[(seed, method)], positions)
        for baseline in paired:
            bootstrap[baseline] = np.mean(
                np.stack([scores[(seed, "SCQE")] - scores[(seed, baseline)]
                          for seed in runs]), axis=0).tolist()
        bootstrap = {baseline: dict(seed=20261013, n=2000,
                                    lower=float(np.percentile(values, 2.5)),
                                    upper=float(np.percentile(values, 97.5)),
                                    conditional_on="five fixed support episodes",
                                    evaluator="cached per-parent COCO matches, exact global accumulate",
                                    raw_differences=values)
                     for baseline, values in bootstrap.items()}
    args.out.mkdir(parents=True)
    with open(args.out / "per_support.csv", "w", newline="") as f:
        columns = [k for k in rows[0] if k != "per_class_AP"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows)
    metrics_path = args.out / "metrics.json"
    temporary = args.out / "metrics.json.tmp"
    temporary.write_text(json.dumps(
        dict(per_support=rows, summary=summary, paired_delta_pp=paired,
             bootstrap=bootstrap, gt_sha256=digest(args.gt)), indent=2))
    os.replace(temporary, metrics_path)


if __name__ == "__main__":
    main()
