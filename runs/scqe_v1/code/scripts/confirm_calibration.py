"""Report frozen 60-image confirmation AP; never retune selected parameters."""
import argparse
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
        raise ValueError("D confirmation uses the frozen three support seeds")
    rows = []
    lock_hashes = set()
    search_checks = {}
    for seed, path in sorted(runs.items()):
        launch = json.loads((path / "launch.json").read_text())
        meta = json.loads((path / "metadata.json").read_text())
        worker = json.loads((path / "worker_done.json").read_text())
        if launch["status"] != "COMPLETE" or meta["expected_queries"] != 60:
            raise RuntimeError(f"incomplete D_check run: {path}")
        if worker.get("deterministic_algorithms") is not True or \
                worker.get("attention") != "DINOv2 standard GPU attention":
            raise RuntimeError(f"confirmation backend policy mismatch: {path}")
        producer = json.loads((path / "producer_done.json").read_text())
        search_checks[seed] = producer.get("no_risk_search_cache_verified")
        if search_checks[seed] is not None and search_checks[seed] < 3:
            raise RuntimeError(f"B2/B3 GPU search cache was not verified: {path}")
        b2_plans = sorted((path / "plans").glob("*__B2.json"))
        if len(b2_plans) != 60:
            raise RuntimeError(f"missing B2 plans: {path}")
        for b2_path in b2_plans:
            b3_path = path / "plans" / b2_path.name.replace("__B2.json", "__B3.json")
            b2, b3 = json.loads(b2_path.read_text()), json.loads(b3_path.read_text())
            if any(b2[key] != b3[key] for key in
                   ("Astar", "F0", "Fplan", "steps", "planned_m")):
                raise RuntimeError(f"B2/B3 pre-gate plans differ: {b2_path.name}")
        lock_hashes.add(meta["lock_sha256"])
        for method in ("B0", "B1", "B2", "B3", "SCQE"):
            metric = evaluate(json.loads((path / f"predictions_{method}.json").read_text()),
                              args.gt)
            rows.append(dict(seed=seed, method=method, **metric))
    if len(lock_hashes) != 1:
        raise RuntimeError("confirmation supports used different parameter locks")
    mean_paired = {}
    for method in ("B1", "B2", "B3", "SCQE"):
        mean_paired[method] = statistics.mean(
            next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == method) -
            next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == "B0")
            for seed in runs)
    args.out.write_text(json.dumps(dict(status="COMPLETE", AP=rows,
                                        mean_paired_delta_pp=mean_paired,
                                        search_cache_verified_hits=search_checks,
                                        lock_hash=next(iter(lock_hashes)),
                                        note="confirmation only; no hyperparameter selection"), indent=2))


if __name__ == "__main__":
    main()
