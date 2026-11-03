"""Evaluate frozen B1/B2/B3 lock on Dcheck60 without retuning."""
import argparse
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="seed:path")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    runs = {}
    for item in args.run:
        seed, path = item.split(":", 1)
        runs[int(seed)] = Path(path)
    if set(runs) != {2023, 42, 1234}:
        raise ValueError("Dcheck60 uses seeds 2023, 42, 1234")
    rows, lock_hashes = [], set()
    for seed, path in sorted(runs.items()):
        launch = json.loads((path / "launch.json").read_text())
        meta = json.loads((path / "metadata.json").read_text())
        if launch["status"] != "COMPLETE" or meta["expected_queries"] != 60 or \
                meta["methods"] != ["B0", "B1", "B2", "B3"]:
            raise RuntimeError(f"incomplete Dcheck60 run: {path}")
        lock_hashes.add(meta["lock_sha256"])
        for method in ("B0", "B1", "B2", "B3"):
            with contextlib.redirect_stdout(io.StringIO()):
                metric = evaluate(json.loads((path / f"predictions_{method}.json").read_text()),
                                  args.gt)
            rows.append(dict(seed=seed, method=method, **metric))
    if len(lock_hashes) != 1:
        raise RuntimeError("Dcheck60 supports used different locks")
    paired = {method: statistics.mean(
        next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == method) -
        next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == "B0")
        for seed in runs) for method in ("B1", "B2", "B3")}
    args.out.write_text(json.dumps(dict(status="COMPLETE", AP=rows,
                                        mean_paired_delta_pp=paired,
                                        lock_hash=next(iter(lock_hashes)),
                                        note="confirmation only; no hyperparameter selection"), indent=2))


if __name__ == "__main__":
    main()
