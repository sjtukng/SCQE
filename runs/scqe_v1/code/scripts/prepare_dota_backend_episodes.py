"""Create write-once CD-ViTO DOTA support episodes from frozen train chips.

Only the selected support instances are exposed to the adaptation backend.
Validation parent annotations and query chips are never read here.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPLITS = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
DATA = ROOT / "data/DOTA-v1.0/chips_1024_gap200/train"
TARGET = ROOT / "c0_datasets/DOTA"
MANIFEST = ROOT / "runs/scqe_v1/backend_dota/episode_inputs.json"
SEEDS = (2023, 42, 1234, 7, 99)
SHOTS = (1, 5, 10)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, document):
    content = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen backend input differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main():
    audit = json.loads((SPLITS / "data_audit.json").read_text())
    integrity = json.loads((SPLITS / "integrity_report.json").read_text())
    if audit.get("status") != "DATA_PREPARED_MODEL_BACKEND_PENDING" or \
            integrity.get("status") != "PASS":
        raise RuntimeError("DOTA frozen data and chip integrity audit must pass first")
    train = json.loads((SPLITS / "train_parents_hbb.json").read_text())
    annotations = {int(row["id"]): row for row in train["annotations"]}
    chips = {row["chip_id"]: row for row in
             json.loads((SPLITS / "train_chips_1024_gap200.json").read_text())["chips"]}
    categories = train["categories"]
    if len(categories) != 15 or [int(row["id"]) for row in categories] != list(range(1, 16)):
        raise RuntimeError("DOTA backend requires the frozen 15-class order")
    link = TARGET / "train"
    TARGET.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != DATA.resolve():
            raise RuntimeError(f"backend train link points to another dataset: {link}")
    elif link.exists():
        raise RuntimeError(f"backend train path exists and is not the frozen link: {link}")
    else:
        link.symlink_to(DATA, target_is_directory=True)
    rows = []
    for seed in SEEDS:
        for shot in SHOTS:
            source = SPLITS / f"dota_support_{shot}shot_seed{seed}.json"
            frozen = json.loads(source.read_text())
            items = frozen["support"]
            counts = Counter(int(item["class_id"]) for item in items)
            if frozen.get("shot") != shot or frozen.get("seed") != seed or \
                    len(items) != 15 * shot or any(counts[c] != shot for c in range(15)):
                raise RuntimeError(f"support budget differs from 15 x {shot}: {source}")
            names = sorted({item["image_id"] for item in items})
            id_by_chip = {name: index for index, name in enumerate(names, 1)}
            images = []
            for name in names:
                chip = chips.get(name)
                if chip is None or chip["split"] != "train" or \
                        not (DATA / (name + ".png")).is_file():
                    raise RuntimeError(f"missing frozen train chip: {name}")
                images.append(dict(id=id_by_chip[name], file_name=name + ".png",
                                   height=1024, width=1024))
            exposed = []
            seen_instances = set()
            for index, item in enumerate(items, 1):
                instance_id = int(item["instance_id"])
                chip = chips[item["image_id"]]
                original = annotations.get(instance_id)
                if instance_id in seen_instances or original is None or \
                        original["difficulty"] != 0 or \
                        int(original["image_id"]) != int(item["parent_id"]) or \
                        int(original["category_id"]) != int(item["class_id"]) + 1 or \
                        int(chip["parent_id"]) != int(item["parent_id"]) or \
                        [chip["x"], chip["y"]] != item["chip_origin_xy"] or \
                        Path(item["image_path"]).absolute() != (DATA / (item["image_id"] + ".png")).absolute():
                    raise RuntimeError(f"support instance/chip mismatch: {instance_id}")
                seen_instances.add(instance_id)
                px, py, pw, ph = map(float, original["bbox"])
                parent_xyxy = [px, py, px + pw, py + ph]
                local_xyxy = [px - chip["x"], py - chip["y"],
                              px + pw - chip["x"], py + ph - chip["y"]]
                if any(abs(a - b) > 1e-6 for a, b in zip(parent_xyxy, item["parent_box_xyxy"])) or \
                        any(abs(a - b) > 1e-6 for a, b in zip(local_xyxy, item["box_xyxy"])) or \
                        not (0 <= local_xyxy[0] < local_xyxy[2] <= chip["content_width"] and
                             0 <= local_xyxy[1] < local_xyxy[3] <= chip["content_height"]):
                    raise RuntimeError(f"support box mismatch: {instance_id}")
                exposed.append(dict(id=index, image_id=id_by_chip[item["image_id"]],
                                    category_id=int(item["class_id"]) + 1,
                                    bbox=[local_xyxy[0], local_xyxy[1], pw, ph],
                                    area=pw * ph, iscrowd=0,
                                    source_instance_id=instance_id))
            target = TARGET / "annotations" / f"{shot}_shot_seed{seed}.json"
            write_once(target, dict(images=images, annotations=exposed,
                                    categories=categories))
            rows.append(dict(seed=seed, shot=shot, support_instances=len(exposed),
                             unique_train_chips=len(images), support_sha256=digest(source),
                             backend_sha256=digest(target), backend_path=str(target)))
    write_once(MANIFEST, dict(status="FROZEN_DOTA_SUPPORT_INPUTS",
                              train_hbb_sha256=digest(SPLITS / "train_parents_hbb.json"),
                              train_chip_map_sha256=digest(SPLITS / "train_chips_1024_gap200.json"),
                              source_integrity_sha256=digest(SPLITS / "integrity_report.json"),
                              episodes=rows))
    print(f"verified {len(rows)} DOTA support episodes: {MANIFEST}")


if __name__ == "__main__":
    main()
