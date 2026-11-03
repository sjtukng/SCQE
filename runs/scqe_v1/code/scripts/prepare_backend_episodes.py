"""Materialize frozen DIOR support episodes as CD-ViTO COCO inputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SUPPORTS = ROOT / "runs/sqr_v1/splits"
SOURCE = ROOT / "data/DIOR/annotations/train.json"
TARGET = ROOT / "c0_datasets/DIOR/annotations"
SEEDS = (2023, 42, 1234, 7, 99)
SHOTS = (1, 5, 10)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train = json.loads(SOURCE.read_text())
    images = {int(x["id"]): x for x in train["images"]}
    anns = {int(x["id"]): x for x in train["annotations"]}
    TARGET.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        for shot in SHOTS:
            source = SUPPORTS / f"dior_support_{shot}shot_seed{seed}.json"
            items = json.loads(source.read_text())["support"]
            counts = Counter(int(x["class_id"]) for x in items)
            if len(items) != 20 * shot or any(counts[c] != shot for c in range(20)):
                raise ValueError(f"invalid support budget: {source}")
            out_images = []
            out_anns = []
            seen_images = set()
            seen_instances = set()
            for index, item in enumerate(items, start=1):
                iid = int(item["instance_id"])
                image_id = int(item["image_id"])
                class_id = int(item["class_id"]) + 1
                original = anns[iid]
                if iid in seen_instances or int(original["image_id"]) != image_id or \
                        int(original["category_id"]) != class_id:
                    raise ValueError(f"support mismatch: {source} instance {iid}")
                seen_instances.add(iid)
                x1, y1, x2, y2 = item["box_xyxy"]
                xywh = [x1, y1, x2 - x1, y2 - y1]
                if any(abs(float(a) - float(b)) > 1e-6
                       for a, b in zip(xywh, original["bbox"])):
                    raise ValueError(f"box mismatch: {source} instance {iid}")
                if image_id not in seen_images:
                    image = images[image_id]
                    out_images.append({k: image[k] for k in ("id", "file_name", "height", "width")})
                    seen_images.add(image_id)
                out_anns.append(dict(id=index, image_id=image_id, category_id=class_id,
                                     bbox=xywh, area=xywh[2] * xywh[3], iscrowd=0))
            doc = dict(images=out_images, annotations=out_anns,
                       categories=train["categories"])
            target = TARGET / f"{shot}_shot_seed{seed}.json"
            if target.exists():
                existing = json.loads(target.read_text())
                if existing != doc:
                    raise ValueError(f"existing backend episode differs: {target}")
                status = "verified_existing"
            else:
                target.write_text(json.dumps(doc))
                status = "created"
            records.append(dict(seed=seed, shot=shot, status=status,
                                support_sha256=digest(source),
                                backend_sha256=digest(target),
                                support_instances=len(items), images=len(out_images)))
    manifest = ROOT / "runs/scqe_v1/backend/episode_inputs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(source_train_sha256=digest(SOURCE),
                                        episodes=records), indent=2))
    print(f"verified {len(records)} frozen episodes; manifest: {manifest}")


if __name__ == "__main__":
    main()
