"""Write an atomic, lightweight progress snapshot for long DIOR Table 1 jobs."""
import datetime
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
SEEDS = (2023, 42, 1234, 7, 99)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def count(directory, suffix):
    if not directory.is_dir():
        return 0
    return sum(name.endswith(suffix) for name in os.listdir(directory))


def complete_producer_queries(path):
    plans = path / "plans"
    if not plans.is_dir():
        return 0
    count_valid = 0
    for plan in plans.glob("*__SCQE.json"):
        qid = plan.name.split("__", 1)[0]
        if not (path / "states" / f"{qid}.pt").is_file():
            continue
        try:
            json.loads(plan.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        count_valid += 1
    return count_valid


def episode(path, expected):
    launch = read(path / "launch.json") or {}
    status = launch.get("status", "NOT_STARTED")
    mode = launch.get("execution_mode", "concurrent_two_gpu") if launch else None
    phase = ("COMPLETE" if status == "COMPLETE" else
             "PROPOSALS" if status == "RUNNING" and
             mode == "sequential_single_gpu" and not (path / "producer_done.json").exists() else
             "CLASSIFICATION" if status == "RUNNING" and
             mode == "sequential_single_gpu" else
             "STREAMING" if status == "RUNNING" else status)
    return dict(status=status, execution_mode=mode, phase=phase,
                expected_queries=expected,
                producer_queries=complete_producer_queries(path),
                classified_jobs=count(path / "detections", ".json"),
                expected_classified_jobs=expected * 5,
                start=launch.get("start"), end=launch.get("end"),
                producer_exit=launch.get("producer_exit"),
                worker_exit=launch.get("worker_exit"))


def snapshot():
    confirmation = {str(seed): episode(RUN / f"confirmation/seed{seed}", 60)
                    for seed in (2023, 42, 1234)}
    formal = {str(shot): {str(seed): episode(
        RUN / "runs/dior_D_5_seed1234_resume2"
        if shot == 5 and seed == 1234 and
        (RUN / "runs/dior_D_5_seed1234_resume2").exists()
        else RUN / "runs/dior_D_5_seed1234_resume1"
        if shot == 5 and seed == 1234 and
        (RUN / "runs/dior_D_5_seed1234_resume1").exists()
        else RUN / f"runs/dior_D_{shot}_seed{seed}", 4900)
                          for seed in SEEDS} for shot in (5, 10, 1)}
    support_ap = {}
    primary = {}
    bootstrap = {}
    for shot in (5, 10, 1):
        support_ap[str(shot)] = {}
        for seed in SEEDS:
            path = RUN / f"metrics/per_support_full/shot{shot}_seed{seed}.json"
            if shot == 5 and not path.is_file():
                path = RUN / f"metrics/per_support_full/seed{seed}.json"
            metric = read(path)
            if metric and metric.get("status") == "COMPLETE":
                support_ap[str(shot)][str(seed)] = {
                    row["method"]: row["AP"] for row in metric["per_support"]}
        p = RUN / f"metrics/dior_D_{shot}_primary/metrics.json"
        b = RUN / f"metrics/dior_D_{shot}/metrics.json"
        primary[str(shot)] = p.is_file()
        bootstrap[str(shot)] = b.is_file()
    failed = [f"confirmation/{seed}" for seed, row in confirmation.items()
              if row["status"] == "FAILED"]
    failed += [f"formal/{shot}/{seed}" for shot, group in formal.items()
               for seed, row in group.items() if row["status"] == "FAILED"]
    return dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                calibration_lock=(RUN / "config_lock_D.json").is_file(),
                confirmation=confirmation, formal=formal,
                support_AP=support_ap, primary_metrics_complete=primary,
                bootstrap_complete=bootstrap, failed_runs=failed)


def main():
    target = RUN / "progress.json"
    while True:
        value = snapshot()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2))
        os.replace(temporary, target)
        if value["failed_runs"] or all(value["bootstrap_complete"].values()):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
