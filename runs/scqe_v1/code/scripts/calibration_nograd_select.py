"""Choose B1/B2/B3 parameters on Dcal40, preserving backend repeat variation."""
import argparse
import contextlib
import hashlib
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    trials = json.loads(args.trials.read_text())
    scales = json.loads(args.scales.read_text())
    by_trial = defaultdict(list)
    baseline_ap = defaultdict(list)
    common = defaultdict(set)
    for item in trials:
        run = Path(item["run_dir"])
        launch = json.loads((run / "launch.json").read_text())
        meta = json.loads((run / "metadata.json").read_text())
        method, seed = item["method"], item["seed"]
        if launch["status"] != "COMPLETE" or meta["expected_queries"] != 40 or \
                meta["methods"] != ["B0", method] or meta["seed"] != seed:
            raise RuntimeError(f"incomplete Dcal40 trial: {run}")
        for name in ("B0", method):
            if len(list((run / "plans").glob(f"*__{name}.json"))) != 40:
                raise RuntimeError(f"missing {name} plans: {run}")
        for field in ("split_sha256", "checkpoint_sha256", "support_sha256",
                      "backend_checkpoint_sha256", "backend_prototype_sha256"):
            common[(seed, field)].add(meta[field])
        baseline_path = run / "predictions_B0.json"
        prediction_path = run / f"predictions_{method}.json"
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = evaluate(json.loads(baseline_path.read_text()), args.gt)
            candidate = evaluate(json.loads(prediction_path.read_text()), args.gt)
        if baseline["n_img"] != 40 or candidate["n_img"] != 40 or \
                baseline["AP"] is None or candidate["AP"] is None:
            raise RuntimeError(f"undefined Dcal40 AP: {run}")
        baseline_ap[seed].append(baseline["AP"])
        by_trial[item["trial_id"]].append(dict(
            seed=seed, method=method, config=item["config"],
            AP=candidate["AP"], B0_AP=baseline["AP"],
            paired_delta=candidate["AP"] - baseline["AP"],
            prediction_sha256=digest(prediction_path),
            baseline_prediction_sha256=digest(baseline_path)))
    if any(len(hashes) != 1 for hashes in common.values()):
        raise RuntimeError("Dcal40 trials used differing split/model/support states")
    candidates = []
    for trial_id, pair in by_trial.items():
        if len(pair) != 2 or {r["seed"] for r in pair} != {2023, 42}:
            raise RuntimeError(f"trial has incomplete supports: {trial_id}")
        candidates.append(dict(trial_id=trial_id, method=pair[0]["method"],
                               config=pair[0]["config"],
                               mean_paired_delta=statistics.mean(x["paired_delta"] for x in pair),
                               support_results=pair))
    selections = {"B0": {}}
    for method in ("B1", "B2", "B3"):
        options = [x for x in candidates if x["method"] == method]
        if not options:
            raise RuntimeError(f"no Dcal40 candidates for {method}")
        tie = (lambda x: x["config"]["budget"]) if method in ("B1", "B2") else \
              (lambda x: x["config"]["eps_true"])
        chosen = min(options, key=lambda x: (-x["mean_paired_delta"], tie(x)))
        selections[method] = dict(chosen["config"],
                                  calibration_trial_id=chosen["trial_id"])
    variation = {str(seed): dict(min=min(values), max=max(values),
                                 range=max(values) - min(values))
                 for seed, values in baseline_ap.items()}
    lock = dict(protocol="D", dataset="DIOR", status="LOCKED_NOGRAD",
                role="B0/B1/B2/B3 only; SCQE derivative E2 failed",
                delta_F=scales["delta_F"], selections=selections,
                calibration_images=40, support_seeds=[2023, 42],
                supervision="DIOR target development labels; protocol D only",
                measured_scales=str(args.scales),
                candidate_results=str(args.out.with_name("calibration_nograd_candidates.json")),
                backend_B0_AP_variation_pp=variation)
    args.out.write_text(json.dumps(lock, indent=2))
    args.out.with_name("calibration_nograd_candidates.json").write_text(
        json.dumps(candidates, indent=2))
    print("selected B1/B2/B3 parameters on frozen Dcal40")


if __name__ == "__main__":
    main()
