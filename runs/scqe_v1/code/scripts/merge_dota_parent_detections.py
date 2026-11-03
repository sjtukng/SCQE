"""Map raw DOTA chip detections to parents; apply one global NMS and cap.

The input must contain class scores before any per-chip threshold, NMS or cap.
This script never reads DOTA validation labels or PF proposal scores.
"""
import argparse
import gzip
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCORE_THRESHOLD = 0.001  # frozen CD-ViTO DIOR ROI_HEADS.SCORE_THRESH_TEST
NMS_IOU = 0.5           # frozen Detectron2 ROI_HEADS.NMS_THRESH_TEST
FINAL_CAP = 100         # frozen TEST.DETECTIONS_PER_IMAGE, applied per parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_chip_map(path):
    data = json.loads(path.read_text())
    chips = {row["chip_id"]: row for row in data["chips"]}
    if len(chips) != len(data["chips"]):
        raise ValueError("duplicate chip ID")
    sizes = {}
    for chip in chips.values():
        parent = int(chip["parent_id"])
        w = int(chip["x"]) + int(chip["content_width"])
        h = int(chip["y"]) + int(chip["content_height"])
        old_w, old_h = sizes.get(parent, (0, 0))
        sizes[parent] = max(old_w, w), max(old_h, h)
        if chip["split"] != "val" or int(chip["width"]) != 1024 or \
                int(chip["height"]) != 1024:
            raise ValueError(f"unexpected chip geometry: {chip['chip_id']}")
    return chips, sizes


def map_one(row, chip, ordinal):
    score = float(row["score"])
    category = int(row["category_id"])
    if not math.isfinite(score) or not 0 <= score <= 1 or not 1 <= category <= 15:
        raise ValueError(f"invalid class score/category at raw row {ordinal}")
    if score < SCORE_THRESHOLD:
        return None
    x, y, w, h = map(float, row["bbox"])
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError(f"invalid raw bbox at row {ordinal}")
    if w <= 0 or h <= 0:
        return None  # invalid class regressor output
    content_w, content_h = int(chip["content_width"]), int(chip["content_height"])
    x1 = min(max(x, 0.0), content_w)
    y1 = min(max(y, 0.0), content_h)
    x2 = min(max(x + w, 0.0), content_w)
    y2 = min(max(y + h, 0.0), content_h)
    if x2 <= x1 or y2 <= y1:
        return None  # prediction lies wholly in padding or outside the chip
    return dict(image_id=int(chip["parent_id"]), category_id=category,
                bbox=[x1 + chip["x"], y1 + chip["y"], x2 - x1, y2 - y1],
                score=score, chip_id=chip["chip_id"], ordinal=ordinal)


def class_nms(rows, accelerated=False):
    """Stable greedy NMS across all chips of a parent for one class."""
    ordered = sorted(rows, key=lambda row: (-row["score"], row["chip_id"], row["ordinal"]))
    if not ordered:
        return []
    if accelerated:
        import torch
        from torchvision.ops import nms
        xywh = np.asarray([row["bbox"] for row in ordered], dtype=np.float32)
        boxes = xywh.copy()
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.tensor([row["score"] for row in ordered], dtype=torch.float32)
        selected = nms(torch.from_numpy(boxes), scores, NMS_IOU).tolist()
        return [ordered[i] for i in selected]
    boxes = np.asarray([row["bbox"] for row in ordered], dtype=np.float64)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    remaining = np.arange(len(ordered))
    kept = []
    while remaining.size:
        best = remaining[0]
        kept.append(ordered[int(best)])
        others = remaining[1:]
        if not others.size:
            break
        iw = np.maximum(0, np.minimum(x2[best], x2[others]) - np.maximum(x1[best], x1[others]))
        ih = np.maximum(0, np.minimum(y2[best], y2[others]) - np.maximum(y1[best], y1[others]))
        intersection = iw * ih
        iou = intersection / (area[best] + area[others] - intersection)
        remaining = others[iou <= NMS_IOU]
    return kept


def merge(chips, raw_rows, stream_parents=False, accelerated=False):
    by_parent_class = defaultdict(list)
    counts = Counter()
    out = []
    per_parent = {}
    active_parent = None
    completed_parents = set()

    def finish_parent(parent):
        if parent is None:
            return
        rows = []
        for (_parent, _category), group in by_parent_class.items():
            if _parent != parent:
                raise RuntimeError("mixed DOTA parents in streaming NMS buffer")
            rows.extend(class_nms(group, accelerated=accelerated))
        ordered = sorted(rows, key=lambda row: (-row["score"], row["category_id"],
                                                row["chip_id"], row["ordinal"]))
        if ordered:
            per_parent[str(parent)] = dict(after_global_nms=len(ordered),
                                           after_final_cap=min(FINAL_CAP, len(ordered)))
            out.extend({key: row[key] for key in ("image_id", "category_id", "bbox", "score")}
                       for row in ordered[:FINAL_CAP])
        by_parent_class.clear()

    for ordinal, row in enumerate(raw_rows):
        chip_id = row["chip_id"]
        chip = chips.get(chip_id)
        if chip is None:
            raise ValueError(f"unknown validation chip: {chip_id}")
        parent = int(chip["parent_id"])
        if stream_parents and parent != active_parent:
            finish_parent(active_parent)
            if parent in completed_parents:
                raise RuntimeError("DOTA raw frames revisit a completed parent")
            if active_parent is not None:
                completed_parents.add(active_parent)
            active_parent = parent
        counts["raw"] += 1
        mapped = map_one(row, chip, ordinal)
        if mapped is None:
            counts["threshold_or_padding_removed"] += 1
            continue
        by_parent_class[(mapped["image_id"], mapped["category_id"])].append(mapped)
        counts["mapped_above_threshold"] += 1
    if stream_parents:
        finish_parent(active_parent)
    else:
        by_parent = dict(by_parent_class)
        by_parent_class.clear()
        for parent in sorted({key[0] for key in by_parent}):
            by_parent_class.update({key: value for key, value in by_parent.items()
                                    if key[0] == parent})
            finish_parent(parent)
    counts["after_global_nms"] = sum(row["after_global_nms"] for row in per_parent.values())
    counts["after_final_parent_cap"] = len(out)
    counts["parents_with_predictions"] = len(per_parent)
    counts["parents_affected_by_cap"] = sum(row["after_global_nms"] > FINAL_CAP
                                           for row in per_parent.values())
    for key in ("raw", "threshold_or_padding_removed", "mapped_above_threshold"):
        counts[key] += 0
    return out, dict(counts), per_parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip-map", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--raw-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.audit.exists():
        raise FileExistsError("DOTA parent outputs are write-once")
    policy = json.loads(args.raw_policy.read_text())
    if policy.get("dataset") != "DOTA-HBB" or \
            policy.get("per_chip_score_threshold", "missing") is not None or \
            policy.get("per_chip_nms", "missing") is not False or \
            policy.get("per_chip_cap", "missing") is not None or \
            policy.get("bbox_format") != "xyxy_float32_abs_original_chip" or \
            policy.get("raw_format") != "SCQEDOTA1_gzip_float32_frames" or \
            policy.get("score_source") != "classifier_only" or \
            policy.get("model_class_count") != 15:
        raise RuntimeError("raw chip detections must precede all chip-level postprocessing")
    chips, parent_sizes = load_chip_map(args.chip_map)

    def read_exact(stream, size):
        data = stream.read(size)
        if len(data) != size:
            raise RuntimeError("truncated DOTA raw tensor stream")
        return data

    def records():
        frames = 0
        with gzip.open(args.raw_jsonl, "rb") as stream:
            if read_exact(stream, 9) != b"SCQEDOTA1":
                raise RuntimeError("DOTA raw tensor stream magic mismatch")
            while True:
                header = stream.read(4)
                if not header:
                    break
                if len(header) != 4:
                    raise RuntimeError("truncated DOTA raw frame header")
                name_len, proposals = struct.unpack("<HH", header)
                chip_id = read_exact(stream, name_len).decode("ascii")
                if chip_id not in chips or proposals > 900:
                    raise RuntimeError("invalid DOTA raw frame chip/count")
                scores = np.frombuffer(read_exact(stream, proposals * 15 * 4), dtype="<f4")
                boxes = np.frombuffer(read_exact(stream, proposals * 15 * 4 * 4),
                                      dtype="<f4").reshape(proposals * 15, 4)
                frames += 1
                for category, (score, box) in enumerate(zip(scores, boxes)):
                    x1, y1, x2, y2 = map(float, box)
                    yield dict(chip_id=chip_id, category_id=category % 15 + 1,
                               bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score))
        if frames != len(chips):
            raise RuntimeError(f"DOTA raw tensor stream has {frames} / {len(chips)} frames")

    predictions, counts, per_parent = merge(chips, records(), stream_parents=True,
                                            accelerated=True)
    if counts["raw"] != policy.get("raw_class_detections") or \
            policy.get("chips") != len(chips):
        raise RuntimeError("DOTA raw tensor stream disagrees with classifier policy")
    raw_sha256 = digest(args.raw_jsonl)
    if policy.get("raw_sha256") != raw_sha256:
        raise RuntimeError("DOTA raw tensor stream hash differs from classifier policy")
    if len(chips) != 5297 or len(parent_sizes) != 458:
        raise RuntimeError("DOTA validation chip/parent manifest is incomplete")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, allow_nan=False))
    args.audit.write_text(json.dumps(dict(status="COMPLETE", policy=policy,
                                         chip_map_sha256=digest(args.chip_map),
                                         raw_sha256=raw_sha256,
                                         output_sha256=digest(args.out),
                                         parent_count=len(parent_sizes),
                                         chip_count=len(chips), counts=counts,
                                         per_parent=per_parent), indent=2))
    print(f"mapped {counts['raw']} raw chip detections to {len(predictions)} parent detections")


if __name__ == "__main__":
    main()
