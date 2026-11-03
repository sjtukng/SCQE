"""Compare identical-support B0 predictions replayed on different GPUs."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text())
    replay = json.loads(args.replay.read_text())
    if len(reference) != len(replay):
        raise RuntimeError(f"detection counts differ: {len(reference)} vs {len(replay)}")
    identity_mismatch = 0
    max_box = 0.0
    max_score = 0.0
    for a, b in zip(reference, replay):
        if (a["image_id"], a["category_id"]) != (b["image_id"], b["category_id"]):
            identity_mismatch += 1
        max_box = max(max_box, *(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])))
        max_score = max(max_score, abs(a["score"] - b["score"]))
    passed = identity_mismatch == 0 and max_box <= 1e-3 and max_score <= 1e-4
    result = dict(status="PASS" if passed else "FAIL", n_detections=len(reference),
                  identity_mismatch=identity_mismatch,
                  max_abs_box_difference=max_box,
                  max_abs_score_difference=max_score,
                  reference=str(args.reference), replay=str(args.replay))
    args.out.write_text(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(f"GPU replay differs; inspect {args.out}")
    print(f"GPU replay passed: {len(reference)} detections, "
          f"max box difference {max_box:.3g}")


if __name__ == "__main__":
    main()
