"""Record measured Dcal AP for completed calibration runs without selecting a lock."""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.trials.read_text())
    records = []
    for entry in manifest:
        run = Path(entry["run_dir"])
        launch_path = run / "launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        record = dict(trial_id=entry["trial_id"], method=entry["method"],
                      seed=entry["seed"], config=entry["config"], run_dir=str(run),
                      status=launch.get("status", "PENDING"))
        if record["status"] == "COMPLETE":
            metadata = json.loads((run / "metadata.json").read_text())
            worker = json.loads((run / "worker_done.json").read_text())
            if metadata["expected_queries"] != 40:
                raise RuntimeError(f"wrong query count: {run}")
            if worker.get("deterministic_algorithms") is not True or \
                    worker.get("attention") != "DINOv2 standard GPU attention":
                raise RuntimeError(f"backend policy mismatch: {run}")
            method_path = run / f"predictions_{entry['method']}.json"
            baseline_path = run / "predictions_B0.json"
            with contextlib.redirect_stdout(io.StringIO()):
                metric = evaluate(json.loads(method_path.read_text()), args.gt)
                baseline = evaluate(json.loads(baseline_path.read_text()), args.gt)
            record.update(AP=metric["AP"], B0_AP=baseline["AP"],
                          paired_delta=metric["AP"] - baseline["AP"],
                          prediction_sha256=digest(method_path),
                          baseline_sha256=digest(baseline_path),
                          baseline_reuse_count=worker.get("baseline_reuse_count", 0),
                          split_sha256=metadata["split_sha256"],
                          lock_sha256=metadata["lock_sha256"])
        records.append(record)
    complete = sum(row["status"] == "COMPLETE" for row in records)
    baseline_hashes = {seed: sorted({row["baseline_sha256"] for row in records
                                      if row["status"] == "COMPLETE" and row["seed"] == seed})
                       for seed in (2023, 42)}
    result = dict(status="COMPLETE" if complete == len(records) else "RUNNING",
                  completed_runs=complete, expected_runs=len(records),
                  baseline_consistent=all(len(hashes) <= 1 for hashes in baseline_hashes.values()),
                  baseline_hashes=baseline_hashes,
                  records=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(args.out)
    print(f"recorded {complete}/{len(records)} completed Dcal runs")


if __name__ == "__main__":
    main()
