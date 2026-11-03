"""Select locked D parameters from completed 40-image two-support trial runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = json.loads(args.trials.read_text())
    scales = json.loads(args.scales.read_text())
    by_trial = {}
    baseline_hashes = {2023: set(), 42: set()}
    for row in rows:
        run = Path(row["run_dir"])
        launch = json.loads((run / "launch.json").read_text())
        meta = json.loads((run / "metadata.json").read_text())
        worker = json.loads((run / "worker_done.json").read_text())
        if launch["status"] != "COMPLETE" or meta["expected_queries"] != 40:
            raise RuntimeError(f"incomplete calibration trial: {run}")
        if worker.get("deterministic_algorithms") is not True or \
                worker.get("attention") != "DINOv2 standard GPU attention":
            raise RuntimeError(f"calibration backend policy mismatch: {run}")
        method, seed = row["method"], row["seed"]
        for m in ("B0", method):
            count = len(list((run / "plans").glob(f"*__{m}.json")))
            if count != 40:
                raise RuntimeError(f"{run} has only {count} plans for {m}")
        b0_path = run / "predictions_B0.json"
        baseline_hashes[seed].add(sha_bytes(b0_path.read_bytes()))
        prediction = run / f"predictions_{method}.json"
        metric = evaluate(json.loads(prediction.read_text()), args.gt)
        base_metric = evaluate(json.loads(b0_path.read_text()), args.gt)
        if metric["AP"] is None or base_metric["AP"] is None:
            raise RuntimeError(f"undefined AP: {run}")
        by_trial.setdefault(row["trial_id"], []).append(dict(
            method=method, seed=seed, config=row["config"], AP=metric["AP"],
            B0_AP=base_metric["AP"], paired_delta=metric["AP"] - base_metric["AP"],
            prediction_sha256=sha_bytes(prediction.read_bytes())))
    if any(len(hashes) != 1 for hashes in baseline_hashes.values()):
        raise RuntimeError("B0 predictions changed between calibration trials")
    candidate_rows = []
    for trial_id, pair in by_trial.items():
        if len(pair) != 2 or {r["seed"] for r in pair} != {2023, 42}:
            raise RuntimeError(f"trial missing a support: {trial_id}")
        method, cfg = pair[0]["method"], pair[0]["config"]
        mean_delta = sum(r["paired_delta"] for r in pair) / 2
        candidate_rows.append(dict(trial_id=trial_id, method=method,
                                   config=cfg, mean_paired_delta=mean_delta,
                                   support_results=pair))
    selections = {}
    for method in ("B1", "B2", "B3", "SCQE", "A3-Cal"):
        options = [r for r in candidate_rows if r["method"] == method]
        if method in ("B1", "B2"):
            key = lambda r: (-r["mean_paired_delta"], r["config"]["budget"])
        elif method == "B3":
            key = lambda r: (-r["mean_paired_delta"], r["config"]["eps_true"])
        else:
            key = lambda r: (-r["mean_paired_delta"], r["config"]["eps_true"],
                             r["config"]["eps_proxy"])
        chosen = min(options, key=key)
        selections[method] = chosen["config"]
        selections[method]["calibration_trial_id"] = chosen["trial_id"]
    selections["B0"] = {}
    lock = dict(protocol="D", dataset="DIOR", status="LOCKED",
                delta_F=scales["delta_F"], selections=selections,
                calibration_images=40, support_seeds=[2023, 42],
                supervision="DIOR target development labels; protocol D only",
                measured_scales=str(args.scales),
                candidate_results=str(args.out.with_name("calibration_candidates.json")))
    args.out.write_text(json.dumps(lock, indent=2))
    args.out.with_name("calibration_candidates.json").write_text(
        json.dumps(candidate_rows, indent=2))


if __name__ == "__main__":
    main()
