#!/usr/bin/env python3
"""SCQE E0: verify frozen SQR v1 DIOR protocol files and generate SCQE v1 splits.

Read-only inputs:
  runs/sqr_v1/splits/dior_dev_queries.json            (100 dev query images)
  runs/sqr_v1/splits/dior_formal_test_queries.json    (4900 formal test queries)
  runs/sqr_v1/splits/dior_support_{1,5,10}shot_seed{S}.json  (S in 2023,42,1234,7,99)
  data/DIOR/annotations/{train,test}.json

Outputs (created only if absent, unless --force):
  runs/scqe_v1/splits/e0_verification.json
  runs/scqe_v1/splits/dior_dcal40.json      (D_cal, original dev-list order)
  runs/scqe_v1/splits/dior_dcheck60.json    (D_check, original dev-list order)
  runs/scqe_v1/splits/dior_final_queries.json (4900 IDs + image_path + width/height)
  runs/scqe_v1/splits/dcal_split_rng_order.json (reproducibility record)

Selection of D_cal/D_check uses ONLY image_id values (no model scores, no GT
difficulty): numpy RandomState(20261012) shuffles a copy of the sorted dev id
list; first 40 -> D_cal, remaining 60 -> D_check.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SQR_SPLITS = os.path.join(ROOT, "runs", "sqr_v1", "splits")
OUT_SPLITS = os.path.join(ROOT, "runs", "scqe_v1", "splits")
TRAIN_JSON = os.path.join(ROOT, "data", "DIOR", "annotations", "train.json")
TEST_JSON = os.path.join(ROOT, "data", "DIOR", "annotations", "test.json")

SEEDS = [2023, 42, 1234, 7, 99]
SHOTS = [1, 5, 10]
SPLIT_RNG_SEED = 20261012
N_CLASSES = 20
D_CAL_N = 40


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inst_key(item):
    """Comparison key per protocol: (image_id, class_id, box) tuple."""
    return (item["image_id"], item["class_id"], tuple(item["box_xyxy"]))


def verify(train, test, test_images_by_id):
    dev = load_json(os.path.join(SQR_SPLITS, "dior_dev_queries.json"))
    formal = load_json(os.path.join(SQR_SPLITS, "dior_formal_test_queries.json"))
    dev_ids = [e["image_id"] for e in dev["dev_queries"]]
    formal_ids = [e["image_id"] for e in formal["formal_test_queries"]]
    dev_set = set(dev_ids)
    test_img_ids = set(test_images_by_id.keys())
    train_img_ids = set(i["id"] for i in train["images"])

    results = {
        "generated_by": "runs/scqe_v1/code/prepare_scqe_splits.py",
        "note": "Read-only verification of frozen runs/sqr_v1/splits protocol files. "
                "No model scores or GT difficulty used for any selection.",
        "a_formal_test_complement": {},
        "b_support_files": {},
        "c_dev_membership": {},
        "overall_pass": None,
    }

    # ---- (a) formal test = test split minus dev 100 ----
    a = {
        "dev_count": len(dev_ids),
        "dev_unique": len(dev_set),
        "formal_count": len(formal_ids),
        "formal_unique": len(set(formal_ids)),
        "test_split_count": len(test_img_ids),
        "expected_complement": len(test_img_ids - dev_set),
        "formal_equals_test_minus_dev": None,
        "formal_ids_not_in_test": None,
        "missing_from_formal": None,
        "dev_ids_duplicated_in_dev_list": sorted([i for i in dev_set if dev_ids.count(i) > 1]),
    }
    a["formal_equals_test_minus_dev"] = (
        a["formal_unique"] == len(formal_ids)
        and set(formal_ids) == test_img_ids - dev_set
        and len(formal_ids) == 4900
    )
    a["formal_ids_not_in_test"] = sorted(set(formal_ids) - test_img_ids)
    a["missing_from_formal"] = sorted((test_img_ids - dev_set) - set(formal_ids))
    a["pass"] = (
        a["formal_equals_test_minus_dev"]
        and not a["formal_ids_not_in_test"]
        and not a["missing_from_formal"]
        and not a["dev_ids_duplicated_in_dev_list"]
        and a["dev_unique"] == 100
    )
    results["a_formal_test_complement"] = a

    # ---- (b) support files per seed ----
    train_anns_by_img_cat = {}
    for ann in train["annotations"]:
        train_anns_by_img_cat.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    all_support_img_ids = set()
    for seed in SEEDS:
        sres = {"seed": seed, "shots": {}, "pass": True}
        per_shot = {}
        for k in SHOTS:
            path = os.path.join(SQR_SPLITS, f"dior_support_{k}shot_seed{seed}.json")
            exists = os.path.isfile(path)
            kres = {"file": os.path.relpath(path, ROOT), "exists": exists}
            if exists:
                data = load_json(path)
                sup = data["support"]
                keys = [inst_key(x) for x in sup]
                per_class = {}
                for x in sup:
                    per_class.setdefault(x["class_id"], []).append(inst_key(x))
                counts = {c: len(v) for c, v in sorted(per_class.items())}
                kres.update({
                    "declared_seed": data.get("seed"),
                    "declared_shot": data.get("shot"),
                    "n_items": len(sup),
                    "n_unique_instances": len(set(keys)),
                    "classes_covered": sorted(per_class.keys()),
                    "per_class_counts": counts,
                    "expected_items": N_CLASSES * k,
                    "exactly_k_unique_per_class": (
                        sorted(per_class.keys()) == list(range(N_CLASSES))
                        and all(len(set(v)) == k and len(v) == k for v in per_class.values())
                    ),
                })
                kres["tuples"] = keys  # kept in memory for nesting checks; not serialized
            sres["shots"][str(k)] = kres
        if all(sres["shots"][str(k)]["exists"] for k in SHOTS):
            def per_class(items):
                d = {}
                for x in items:
                    d.setdefault(x["class_id"], []).append(inst_key(x))
                return d
            pc = {k: per_class(load_json(os.path.join(
                SQR_SPLITS, f"dior_support_{k}shot_seed{seed}.json"))["support"]) for k in SHOTS}
            sres["nesting_1_in_5"] = all(pc[1][c] == pc[5][c][:1] for c in range(N_CLASSES))
            sres["nesting_5_in_10"] = all(pc[5][c] == pc[10][c][:5] for c in range(N_CLASSES))
            imgs = {x[0] for x in pc[10].values() for x in x}
            all_support_img_ids |= imgs
            sres["unique_support_images"] = len(imgs)
            sres["all_support_imgs_in_train"] = imgs <= train_img_ids
            sres["support_imgs_not_in_train"] = sorted(imgs - train_img_ids)
            sres["overlap_with_dev"] = sorted(imgs & dev_set)
            sres["overlap_with_test_split"] = sorted(imgs & test_img_ids)
            mismatch = []
            data10 = load_json(os.path.join(SQR_SPLITS, f"dior_support_10shot_seed{seed}.json"))
            for x in data10["support"]:
                anns = train_anns_by_img_cat.get((x["image_id"], x["class_id"] + 1), [])
                bx = [float(v) for v in x["box_xyxy"]]
                hit = any(
                    abs(a["bbox"][0] - bx[0]) < 1e-3 and abs(a["bbox"][1] - bx[1]) < 1e-3
                    and abs(a["bbox"][0] + a["bbox"][2] - bx[2]) < 1e-3
                    and abs(a["bbox"][1] + a["bbox"][3] - bx[3]) < 1e-3
                    for a in anns
                )
                if not hit:
                    mismatch.append(inst_key(x))
            sres["boxes_match_train_annotations"] = not mismatch
            sres["box_mismatches"] = mismatch
            sres["pass"] = (
                sres["nesting_1_in_5"]
                and sres["nesting_5_in_10"]
                and all(sres["shots"][str(k)]["exactly_k_unique_per_class"] for k in SHOTS)
                and all(sres["shots"][str(k)]["declared_seed"] == seed and sres["shots"][str(k)]["declared_shot"] == k for k in SHOTS)
                and all(sres["shots"][str(k)]["n_items"] == N_CLASSES * k for k in SHOTS)
                and sres["all_support_imgs_in_train"]
                and not sres["overlap_with_dev"]
                and not sres["overlap_with_test_split"]
                and sres["boxes_match_train_annotations"]
            )
        else:
            sres["pass"] = False
        for k in SHOTS:
            sres["shots"][str(k)].pop("tuples", None)
        results["b_support_files"][str(seed)] = sres

    # ---- (c) dev 100 membership ----
    dev_not_in_test = sorted(dev_set - test_img_ids)
    dev_img_paths_ok = all(
        e["image_path"] == f"test/new_test/{e['image_id']:05d}.jpg" for e in dev["dev_queries"]
    )
    c = {
        "dev_ids_in_test_json": not dev_not_in_test,
        "dev_ids_not_in_test_json": dev_not_in_test,
        "dev_image_path_pattern_ok": dev_img_paths_ok,
        "overlap_dev_with_support_images": sorted(dev_set & all_support_img_ids),
        "pass": not dev_not_in_test and not (dev_set & all_support_img_ids) and dev_img_paths_ok,
    }
    results["c_dev_membership"] = c

    results["overall_pass"] = (
        a["pass"]
        and all(s["pass"] for s in results["b_support_files"].values())
        and c["pass"]
    )
    return dev, formal, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files")
    args = ap.parse_args()

    os.makedirs(OUT_SPLITS, exist_ok=True)

    train = load_json(TRAIN_JSON)
    test = load_json(TEST_JSON)
    test_images_by_id = {i["id"]: i for i in test["images"]}

    dev, formal, results = verify(train, test, test_images_by_id)

    # ---------- D_cal / D_check split ----------
    dev_ids = [e["image_id"] for e in dev["dev_queries"]]
    rng = np.random.RandomState(SPLIT_RNG_SEED)
    shuffled = list(sorted(dev_ids))
    rng.shuffle(shuffled)
    dcal_ids = set(shuffled[:D_CAL_N])
    dcheck_ids = set(shuffled[D_CAL_N:])
    assert len(dcal_ids) == 40 and len(dcheck_ids) == 60 and not (dcal_ids & dcheck_ids)

    # preserve ORIGINAL dev-list order
    dcal = [e for e in dev["dev_queries"] if e["image_id"] in dcal_ids]
    dcheck = [e for e in dev["dev_queries"] if e["image_id"] in dcheck_ids]

    # final queries: 4900, IDs + image_path + width/height (width/height from test.json)
    formal_ids_seen = set()
    final_queries = []
    for e in formal["formal_test_queries"]:
        iid = e["image_id"]
        if iid in formal_ids_seen:
            continue
        formal_ids_seen.add(iid)
        img = test_images_by_id[iid]
        final_queries.append({
            "image_id": iid,
            "image_path": e["image_path"],
            "width": img["width"],
            "height": img["height"],
        })

    common_meta = {
        "source": "runs/sqr_v1/splits (frozen, read-only)",
        "split_rng_seed": SPLIT_RNG_SEED,
        "selection_criterion": "image_id only; numpy RandomState(%d) shuffle of sorted dev id list; "
                               "first 40 -> D_cal, rest 60 -> D_check; no model scores / GT difficulty used" % SPLIT_RNG_SEED,
        "protocol_sha256": {
            "dior_dev_queries.json": sha256_of(os.path.join(SQR_SPLITS, "dior_dev_queries.json")),
            "dior_formal_test_queries.json": sha256_of(os.path.join(SQR_SPLITS, "dior_formal_test_queries.json")),
        },
    }

    outputs = {
        "e0_verification.json": results,
        "dior_dcal40.json": {"meta": dict(common_meta, subset="D_cal", count=len(dcal)), "queries": dcal},
        "dior_dcheck60.json": {"meta": dict(common_meta, subset="D_check", count=len(dcheck)), "queries": dcheck},
        "dior_final_queries.json": {
            "meta": dict(common_meta, subset="formal_test_final_4900",
                         note="width/height taken from data/DIOR/annotations/test.json; no annotations included"),
            "queries": final_queries,
        },
        "dcal_split_rng_order.json": {
            "meta": dict(common_meta, note="full shuffled order of the 100 sorted dev ids; "
                                           "positions 0-39 = D_cal, 40-99 = D_check"),
            "rng": "numpy.random.RandomState(%d)" % SPLIT_RNG_SEED,
            "shuffled_dev_ids": shuffled,
            "dcal_ids": shuffled[:D_CAL_N],
            "dcheck_ids": shuffled[D_CAL_N:],
        },
    }

    for name, payload in outputs.items():
        path = os.path.join(OUT_SPLITS, name)
        if os.path.exists(path) and not args.force:
            print(f"[skip] {path} exists (use --force to overwrite)")
            continue
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[write] {path}")

    # cross-check: existing verification file on disk should match if rerun
    print("\n=== E0 verification summary ===")
    print("overall_pass:", results["overall_pass"])
    print("(a) formal test = 4900 complement of dev 100:", results["a_formal_test_complement"]["pass"])
    for seed in SEEDS:
        s = results["b_support_files"][str(seed)]
        print(f"(b) seed {seed}: pass={s['pass']} nesting 1⊂5={s.get('nesting_1_in_5')} 5⊂10={s.get('nesting_5_in_10')}")
    print("(c) dev membership:", results["c_dev_membership"]["pass"])
    if not results["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
