"""DIOR protocol data access (frozen scqe_v1 splits; no GT used online)."""
import json
from pathlib import Path
from typing import Dict, List

from .config import DIOR_DIR, RUN_DIR


def load_json(p):
    with open(p) as f:
        return json.load(f)


def _entries(data):
    if isinstance(data, dict):
        for key in ("queries", "images", "entries", "dev_queries"):
            if key in data:
                return data[key]
    return data


def dcal_entries() -> List[Dict]:
    return _entries(load_json(RUN_DIR / "splits" / "dior_dcal40.json"))


def dcheck_entries() -> List[Dict]:
    return _entries(load_json(RUN_DIR / "splits" / "dior_dcheck60.json"))


def support_instances(seed: int, shot: int) -> List[Dict]:
    """Support episode: C x K selected instances (train split only)."""
    data = load_json(RUN_DIR.parent / "sqr_v1" / "splits" /
                     f"dior_support_{shot}shot_seed{seed}.json")
    items = data["support"]
    out = []
    for it in items:
        out.append(dict(
            class_id=int(it["class_id"]),
            image_id=int(it["image_id"]),
            image_path=str(DIOR_DIR / "images" / it["image_path"]),
            box_xyxy=[float(v) for v in it["box_xyxy"]],
            ann_id=it.get("instance_id"),
        ))
    return out


def query_image_path(entry: Dict) -> str:
    return str(DIOR_DIR / "images" / entry["image_path"])
