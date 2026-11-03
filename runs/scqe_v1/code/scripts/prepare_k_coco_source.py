#!/usr/bin/env python3
"""Freeze an independent, non-target COCO-2017 development source for protocol K.

This script uses only COCO val2017 source labels. It never reads DIOR/DOTA query
annotations or model predictions. The nine animal categories are disjoint from
the named DIOR and DOTA detection categories. Outputs are write-once.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "data/COCO2017_K/annotations/instances_val2017.json"
IMAGES = ROOT / "data/COCO2017_K/val2017"
OUT = ROOT / "runs/scqe_v1/splits/k_coco2017"
CLASS_NAMES = ("cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
SEEDS = (2023, 42, 1234)
SPLIT_SEED = 20261017


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--images", type=Path, default=IMAGES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    source = json.loads(args.source.read_text())
    cats = {x["name"]: x for x in source["categories"]}
    selected_ids = {cats[name]["id"] for name in CLASS_NAMES}
    selected_cats = [cats[name] for name in CLASS_NAMES]
    image_by_id = {int(x["id"]): x for x in source["images"]}
    if len(image_by_id) != 5000:
        raise RuntimeError("expected 5000 COCO val2017 images")
    ann_by_img = defaultdict(list)
    images_by_cat = defaultdict(set)
    for ann in source["annotations"]:
        if ann["category_id"] not in selected_ids or ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        iid = int(ann["image_id"])
        ann_by_img[iid].append(ann)
        images_by_cat[ann["category_id"]].add(iid)

    rng = random.Random(SPLIT_SEED)
    cal, check = set(), set()
    # Source-label stratification guarantees every calibration/check category is
    # represented. No model score, target label or difficulty is consulted.
    for cat in selected_cats:
        cid = cat["id"]
        for target, wanted, forbidden in ((cal, 4, check), (check, 6, cal)):
            have = len(target & images_by_cat[cid])
            candidates = sorted(images_by_cat[cid] - cal - check)
            rng.shuffle(candidates)
            if have + len(candidates) < wanted:
                raise RuntimeError(f"insufficient source images for {cat['name']}")
            target.update(candidates[:max(0, wanted - have)])
            assert not (target & forbidden)
    if len(cal) > 40 or len(check) > 60:
        raise RuntimeError("source stratification exceeds 40/60 budget")
    empty = [i for i in sorted(image_by_id) if i not in cal | check and not ann_by_img[i]]
    rng.shuffle(empty)
    cal_fill = 40 - len(cal)
    check_fill = 60 - len(check)
    cal.update(empty[:cal_fill])
    check.update(empty[cal_fill:cal_fill + check_fill])
    if len(cal) != 40 or len(check) != 60 or cal & check:
        raise RuntimeError("could not form disjoint 40/60 split")

    query = cal | check
    pool_by_cat = defaultdict(list)
    for iid, anns in ann_by_img.items():
        if iid not in query:
            for ann in anns:
                pool_by_cat[ann["category_id"]].append(ann)
    support_docs = {}
    support_images = set()
    for seed in SEEDS:
        srng = random.Random(seed)
        chosen = []
        for cat in selected_cats:
            options = sorted(pool_by_cat[cat["id"]], key=lambda a: int(a["id"]))
            srng.shuffle(options)
            if len(options) < 5:
                raise RuntimeError(f"insufficient source supports for {cat['name']}")
            for ann in options[:5]:
                iid = int(ann["image_id"])
                support_images.add(iid)
                chosen.append({
                    "instance_id": int(ann["id"]), "image_id": iid,
                    "class_id": int(ann["category_id"]),
                    "class_name": cat["name"], "box_xywh": ann["bbox"],
                    "image_path": str((args.images / image_by_id[iid]["file_name"]).absolute()),
                })
        support_docs[seed] = {"seed": seed, "shot": 5, "support": chosen}
    if support_images & query:
        raise RuntimeError("source support/query image overlap")

    all_used = query | support_images
    missing = [i for i in sorted(all_used) if not (args.images / image_by_id[i]["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected COCO val images absent; first IDs: {missing[:10]}")

    def image_record(iid):
        original = image_by_id[iid]
        return {"image_id": iid, "image_path": str((args.images / original["file_name"]).absolute()),
                "width": original["width"], "height": original["height"]}

    for name, ids in (("kcal40", cal), ("kcheck60", check)):
        write_once(args.out / f"{name}.json", {
            "source": "COCO 2017 validation / nine non-target animal detection categories",
            "images": [image_record(i) for i in sorted(ids)],
        })
    query_gt = {
        "images": [image_by_id[i] for i in sorted(query)],
        "annotations": sorted((a for i in query for a in ann_by_img[i]), key=lambda a: int(a["id"])),
        "categories": selected_cats,
    }
    write_once(args.out / "k_dev_gt.json", query_gt)
    for seed, doc in support_docs.items():
        write_once(args.out / f"k_source_support_5shot_seed{seed}.json", doc)
    write_once(args.out / "source_audit.json", {
        "status": "SOURCE_FROZEN_BACKEND_NOT_CALIBRATED",
        "source_dataset": "COCO 2017 val2017",
        "source_urls": {
            "images": "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
            "annotations": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "source_annotation_sha256": sha256(args.source),
        "source_images_archive_sha256": sha256(args.images.parent / "downloads/val2017.zip"),
        "source_annotations_archive_sha256": sha256(args.images.parent / "downloads/annotations_trainval2017.zip"),
        "source_images_dir": str(args.images.absolute()),
        "task_exclusion": {
            "classes": list(CLASS_NAMES),
            "relation": "animal object classes; no DIOR/DOTA target category has an animal class name",
            "scope": "category labels, not a guarantee that source photographs contain no target-like background object",
        },
        "source_label_budget": {
            "cal_images": 40, "check_images": 60,
            "query_instances": len(query_gt["annotations"]),
            "support_instances_per_seed": 5 * len(CLASS_NAMES),
            "support_seeds": list(SEEDS),
            "unique_support_image_ids": sorted(support_images),
            "unique_support_instance_ids": sorted({x["instance_id"] for doc in support_docs.values() for x in doc["support"]}),
        },
        "selection": "source class presence for 4/6 per class; fill with source-empty images; fixed RNG 20261017; no predictions or target query labels",
        "cal_image_ids": sorted(cal), "check_image_ids": sorted(check),
        "prior_use_history": "No prior use of COCO val2017 as an SCQE K calibration source found in this workspace before this preparation; external histories not independently auditable.",
        "model_pretraining_note": "PF-RPN checkpoint name indicates COCO pretraining; verify exact train/val exposure for every backend before formal K calibration.",
        "limitations": "A frozen non-target source alone does not supply a K backend adaptation or config_lock_K.json.",
    })
    print(f"frozen K source: cal={len(cal)} check={len(check)} GT={len(query_gt['annotations'])} support images={len(support_images)} at {args.out}")


if __name__ == "__main__":
    main()
