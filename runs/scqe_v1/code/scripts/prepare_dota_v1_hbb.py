#!/usr/bin/env python3
"""Prepare DOTA-v1.0 parent-level HBB inputs and deterministic 1024/200 chips.

Only train parents supply supports; only validation parents supply queries.
Difficulty 0 is eligible for supports; difficulty 0 and 1 remain ordinary GT.
This is a data preparation entrypoint, not a DOTA model or evaluation backend.
"""

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# DOTA includes genuine very large parent PNGs (up to tens of thousands of
# pixels per side), above Pillow's generic decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/DOTA-v1.0"
OUT = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
CLASSES = (
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
)
SEEDS = (2023, 42, 1234, 7, 99)
CHIP_SIZE = 1024
OVERLAP = 200
STRIDE = CHIP_SIZE - OVERLAP


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, doc):
    content = json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def chips_for(parent_id, width, height, split, chip_dir):
    def offsets(dimension):
        values = []
        position = 0
        while position < dimension:
            if position + CHIP_SIZE >= dimension:
                position = max(dimension - CHIP_SIZE, 0)
                if not values or values[-1] != position:
                    values.append(position)
                break
            values.append(position)
            position += STRIDE
        return values

    out = []
    for row, y in enumerate(offsets(height)):
        for col, x in enumerate(offsets(width)):
            name = f"P{parent_id:04d}__r{row:03d}_c{col:03d}.png"
            out.append({
                "chip_id": name[:-4], "parent_id": parent_id, "split": split,
                "row": row, "col": col, "x": x, "y": y,
                "content_width": min(CHIP_SIZE, width - x),
                "content_height": min(CHIP_SIZE, height - y),
                "width": CHIP_SIZE, "height": CHIP_SIZE,
                "image_path": str((chip_dir / split / name).absolute()),
            })
    return out


def parse_parent(split, img_path, label_path, ann_start, chip_dir, decode):
    with Image.open(img_path) as image:
        width, height = image.size
        if decode:
            image.load()
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image dimensions: {img_path}")
    parent_id = int(img_path.stem[1:])
    annotations = []
    discarded = Counter()
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].startswith(("imagesource:", "gsd:", "acquisition")):
            continue
        if len(parts) != 10:
            raise ValueError(f"malformed DOTA annotation {label_path}:{line_no}")
        try:
            obb = [float(x) for x in parts[:8]]
            difficulty = int(parts[9])
        except ValueError as exc:
            raise ValueError(f"malformed DOTA number {label_path}:{line_no}") from exc
        name = parts[8]
        if name not in CLASSES or difficulty not in (0, 1):
            raise ValueError(f"unexpected DOTA category/difficulty {label_path}:{line_no}")
        x1 = max(0.0, min(float(width), min(obb[::2])))
        y1 = max(0.0, min(float(height), min(obb[1::2])))
        x2 = max(0.0, min(float(width), max(obb[::2])))
        y2 = max(0.0, min(float(height), max(obb[1::2])))
        if x2 <= x1 or y2 <= y1:
            discarded["nonpositive_hbb_after_clip"] += 1
            continue
        ann_id = ann_start + len(annotations)
        annotations.append({
            "id": ann_id, "image_id": parent_id, "category_id": CLASSES.index(name) + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            "difficulty": difficulty, "obb_xy": obb,
            "source_line": line_no, "source_file": str(label_path.absolute()),
        })
    image = {"id": parent_id, "file_name": str(img_path.absolute()),
             "width": width, "height": height}
    return image, annotations, chips_for(parent_id, width, height, split, chip_dir), discarded


def _materialize_parent(job):
    source_path, records = job
    created = 0
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        for chip in records:
            dst = Path(chip["image_path"])
            if dst.exists():
                try:
                    with Image.open(dst) as existing:
                        if existing.size != (CHIP_SIZE, CHIP_SIZE):
                            raise ValueError(f"bad existing chip: {dst}")
                        existing.verify()
                    continue
                except (OSError, ValueError, SyntaxError):
                    dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            x, y = chip["x"], chip["y"]
            patch = source.crop((x, y, x + chip["content_width"], y + chip["content_height"]))
            canvas = Image.new("RGB", (CHIP_SIZE, CHIP_SIZE), (0, 0, 0))
            canvas.paste(patch, (0, 0))
            temporary = dst.with_suffix(".png.tmp")
            canvas.save(temporary, format="PNG", compress_level=1)
            temporary.replace(dst)
            created += 1
    return created


def materialize_chips(chips, image_by_id, workers=1):
    by_parent = defaultdict(list)
    for chip in chips:
        by_parent[chip["parent_id"]].append(chip)
    jobs = [(image_by_id[parent_id]["file_name"], records)
            for parent_id, records in sorted(by_parent.items())]
    if workers == 1:
        return sum(_materialize_parent(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_materialize_parent, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--decode-images", action="store_true")
    ap.add_argument("--materialize-chips", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    chip_dir = args.data / "chips_1024_gap200"
    all_images = {}
    all_annotations = {}
    all_chips = {}
    discards = {}
    annotation_hashes = {}
    for split, expected in (("train", 1411), ("val", 458)):
        img_dir = args.data / split / "images"
        txt_dir = args.data / split / "labelTxt"
        image_files = sorted(img_dir.glob("P*.png"))
        label_files = sorted(txt_dir.glob("P*.txt"))
        if len(image_files) != expected or len(label_files) != expected:
            raise RuntimeError(f"{split} image/label count: {len(image_files)}/{len(label_files)}; expected {expected}")
        if {p.stem for p in image_files} != {p.stem for p in label_files}:
            raise RuntimeError(f"{split} image/label parent IDs mismatch")
        images, anns, chips = [], [], []
        discard = Counter()
        for path in image_files:
            image, these_anns, these_chips, these_discard = parse_parent(
                split, path, txt_dir / (path.stem + ".txt"), len(anns) + 1,
                chip_dir, args.decode_images)
            images.append(image)
            anns.extend(these_anns)
            chips.extend(these_chips)
            discard.update(these_discard)
        all_images[split] = images
        all_annotations[split] = anns
        all_chips[split] = chips
        discards[split] = dict(discard)
        archive = args.data / "downloads" / split / "labelTxt-v1.0.zip"
        annotation_hashes[split] = sha256(archive)
        write_once(args.out / f"{split}_parents_hbb.json", {
            "images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
            "protocol": "DOTA-v1.0 OBB vertices to clipped HBB; difficulty 0/1 ordinary GT",
        })
        write_once(args.out / f"{split}_chips_1024_gap200.json", {
            "chip_size": CHIP_SIZE, "overlap": OVERLAP, "stride": STRIDE,
            "edge_policy": "DOTA devkit-style row-major 824 stride with final edge-aligned chip; RGB black pad only when parent dimension < 1024",
            "chips": chips,
        })

    train_chips = defaultdict(list)
    for chip in all_chips["train"]:
        train_chips[chip["parent_id"]].append(chip)
    eligible = defaultdict(list)
    for ann in all_annotations["train"]:
        if ann["difficulty"] != 0:
            continue
        x, y, w, h = ann["bbox"]
        matches = [chip for chip in train_chips[ann["image_id"]]
                   if x >= chip["x"] and y >= chip["y"]
                   and x + w <= chip["x"] + chip["content_width"]
                   and y + h <= chip["y"] + chip["content_height"]]
        if matches:
            eligible[ann["category_id"]].append((ann, matches[0]))
    materialize = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        chosen = []
        for class_id in range(1, len(CLASSES) + 1):
            options = sorted(eligible[class_id], key=lambda p: p[0]["id"])
            rng.shuffle(options)
            if len(options) < 10:
                raise RuntimeError(f"DOTA class {class_id} has only {len(options)} eligible supports")
            for ann, chip in options[:10]:
                x, y, w, h = ann["bbox"]
                chosen.append({
                    "instance_id": ann["id"], "parent_id": ann["image_id"],
                    "image_id": chip["chip_id"], "image_path": chip["image_path"],
                    "class_id": class_id - 1, "class_name": CLASSES[class_id - 1],
                    "box_xyxy": [x - chip["x"], y - chip["y"], x + w - chip["x"], y + h - chip["y"]],
                    "parent_box_xyxy": [x, y, x + w, y + h],
                    "chip_origin_xy": [chip["x"], chip["y"]],
                    "difficulty": 0,
                })
                materialize[chip["chip_id"]] = chip
        for shot in (1, 5, 10):
            selected = [item for cid in range(len(CLASSES))
                        for item in [x for x in chosen if x["class_id"] == cid][:shot]]
            write_once(args.out / f"dota_support_{shot}shot_seed{seed}.json", {
                "seed": seed, "shot": shot, "support": selected,
                "policy": "difficulty=0 only; full HBB inside a chip; first row-major eligible chip",
            })
    train_ids = {x["id"] for x in all_images["train"]}
    val_ids = {x["id"] for x in all_images["val"]}
    if train_ids & val_ids:
        raise RuntimeError("DOTA train/validation parent overlap")
    if args.materialize_chips:
        image_by_id = {x["id"]: x for x in all_images["train"] + all_images["val"]}
        n_train = materialize_chips(list(materialize.values()), image_by_id, args.workers)
        n_val = materialize_chips(all_chips["val"], image_by_id, args.workers)
        print(f"created {n_train} train support chips and {n_val} validation query chips")
    chip_files_present = {
        "train_support": sum(Path(x["image_path"]).is_file() for x in materialize.values()),
        "val_query": sum(Path(x["image_path"]).is_file() for x in all_chips["val"]),
    }
    gt_per_parent = Counter(x["image_id"] for x in all_annotations["val"])
    write_once(args.out / "data_audit.json", {
        "status": "DATA_PREPARED_MODEL_BACKEND_PENDING" if
                  chip_files_present == {"train_support": len(materialize), "val_query": len(all_chips["val"])}
                  else "PARENT_DATA_FROZEN_CHIPS_PENDING",
        "source": "DOTA-v1.0; images mirrored from isaaccorley/dota with published SHA256; original v1.0 OBB labels from official DOTA Drive",
        "official_labels_sha256": annotation_hashes,
        "images_archive_sha256": {
            split: sha256(args.data / "downloads" / split / "images.tar.gz") for split in ("train", "val")
        },
        "parent_counts": {split: len(all_images[split]) for split in ("train", "val")},
        "instance_counts": {split: len(all_annotations[split]) for split in ("train", "val")},
        "chip_counts": {split: len(all_chips[split]) for split in ("train", "val")},
        "chip_files_present": chip_files_present,
        "discarded": discards,
        "val_parents_gt_over_100": sum(n > 100 for n in gt_per_parent.values()),
        "val_parents_empty_gt": len(val_ids) - len(gt_per_parent),
        "support_seeds": list(SEEDS), "shots": [1, 5, 10],
        "support_unique_chips_10shot_all_seeds": len(materialize),
        "rule": "train parents support only, validation parents query only; HBB clipped to parent; positive area; support difficulty0; eval difficulty0/1; exact 1024/200 chips",
    })
    print(f"DOTA data frozen at {args.out}; train/val parents {len(train_ids)}/{len(val_ids)}")


if __name__ == "__main__":
    main()
