"""Freeze score-independent query subsets before diagnostic or timing inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--count", type=int, choices=(20, 50), required=True)
    ap.add_argument("--role", choices=("diagnostic", "timing"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt-source", type=Path)
    ap.add_argument("--gt-out", type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = json.loads(args.source.read_text())
    entries = data.get("queries", data) if isinstance(data, dict) else data
    if len(entries) < args.count:
        raise ValueError("source manifest has too few queries")
    subset = sorted(entries, key=lambda x: str(x["image_id"]))[:args.count]
    result = dict(meta=dict(role=args.role, count=args.count,
                            selection="lexicographic image ID; before inference",
                            source=str(args.source),
                            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest()),
                  queries=subset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    if args.gt_source:
        if args.gt_out is None or args.gt_out.exists():
            raise ValueError("--gt-out must name a new file")
        gt = json.loads(args.gt_source.read_text())
        chosen = {int(q["image_id"]) for q in subset}
        gt_subset = dict(info=gt.get("info", {}), licenses=gt.get("licenses", []),
                         images=[im for im in gt["images"] if int(im["id"]) in chosen],
                         annotations=[ann for ann in gt["annotations"]
                                      if int(ann["image_id"]) in chosen],
                         categories=gt["categories"])
        if len(gt_subset["images"]) != args.count:
            raise ValueError("GT does not contain all selected images")
        args.gt_out.parent.mkdir(parents=True, exist_ok=True)
        args.gt_out.write_text(json.dumps(gt_subset))


if __name__ == "__main__":
    main()
