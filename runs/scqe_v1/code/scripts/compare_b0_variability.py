"""Quantify CD-ViTO inference variation on the fixed DIOR development set.

The backend is empirically variable even on repeated GPU-0 inference with the
same weights and proposals. Compare GPU-1/GPU-0 drift against that intrinsic
GPU-0 repeat, preserving the strict equality check as a separate failed test.
"""
import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def comparison(left, right):
    by_left, by_right = defaultdict(list), defaultdict(list)
    for det in left:
        by_left[(det["image_id"], det["category_id"])].append(det)
    for det in right:
        by_right[(det["image_id"], det["category_id"])].append(det)
    label_count_difference = sum(abs(len(by_left[key]) - len(by_right[key]))
                                 for key in by_left.keys() | by_right.keys())
    box_delta, score_delta = [], []
    for key in by_left.keys() & by_right.keys():
        a, b = by_left[key], by_right[key]
        costs = np.asarray([
            [sum((x - y) ** 2 for x, y in zip(d["bbox"], e["bbox"])) +
             100 * (d["score"] - e["score"]) ** 2 for e in b]
            for d in a])
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            box_delta.append(max(abs(x - y) for x, y in zip(a[i]["bbox"], b[j]["bbox"])))
            score_delta.append(abs(a[i]["score"] - b[j]["score"]))
    return dict(matched=len(box_delta), label_count_difference=label_count_difference,
                box_p95=float(np.percentile(box_delta, 95)),
                box_p99=float(np.percentile(box_delta, 99)),
                score_p95=float(np.percentile(score_delta, 95)),
                score_p99=float(np.percentile(score_delta, 99)))


def main():
    ap = argparse.ArgumentParser()
    for name in ("gpu1", "gpu0_a", "gpu0_b", "gt", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = ap.parse_args()
    paths = {name: getattr(args, name) for name in ("gpu1", "gpu0_a", "gpu0_b")}
    preds = {name: json.loads(path.read_text()) for name, path in paths.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.gt))
        metrics = {name: evaluate(data, gt) for name, data in preds.items()}
    repeat = comparison(preds["gpu0_a"], preds["gpu0_b"])
    cross = comparison(preds["gpu1"], preds["gpu0_a"])
    result = dict(
        purpose="Development-only backend device/repeat diagnostic; no formal AP",
        predictions={name: str(path) for name, path in paths.items()},
        AP={name: metrics[name]["AP"] for name in paths},
        same_device_repeat=repeat, cross_device=cross,
        strict_equality=False,
        cross_within_repeat_envelope=(
            cross["label_count_difference"] <= repeat["label_count_difference"] and
            cross["box_p95"] <= repeat["box_p95"] and
            cross["score_p95"] <= repeat["score_p95"]),
        cross_AP_difference_pp=abs(metrics["gpu1"]["AP"] - metrics["gpu0_a"]["AP"]),
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
