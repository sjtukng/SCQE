"""Freeze all B1/B2/B3 Dcal40 candidates before their AP is observed."""
import argparse
import json
import shlex
from pathlib import Path

LAUNCHER = Path(__file__).with_name("run_dior_pair.py")
PYTHON = Path("/home/root-user/.conda/envs/sqr/bin/python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scales = json.loads(args.scales.read_text())
    if (scales.get("protocol"), scales.get("dataset"), scales.get("images"),
            scales.get("status")) != ("D", "DIOR", 12, "MEASURED"):
        raise ValueError("expected measured non-derivative D scales")
    entries = json.loads(args.split.read_text())["queries"]
    if len(entries) != 40:
        raise ValueError("D calibration requires all 40 frozen images")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "locks").mkdir()
    s_t = float(scales["s_t"])
    true_grid = sorted({min(s_t / 3, 1.0), min(s_t, 1.0), min(3 * s_t, 1.0)})
    trials = [(f"{method}_b{budget}", method, {"budget": budget})
              for method in ("B1", "B2") for budget in (8, 16, 32)]
    trials += [(f"B3_t{i}", "B3", {"eps_true": threshold})
               for i, threshold in enumerate(true_grid)]
    manifest, commands = [], []
    for trial_id, method, config in trials:
        lock = dict(protocol="D", dataset="DIOR",
                    status="CALIBRATION_TRIAL_NOGRAD",
                    delta_F=scales["delta_F"],
                    selections={"B0": {}, method: config},
                    measured_scales=str(args.scales))
        lock_path = args.out / "locks" / f"{trial_id}.json"
        lock_path.write_text(json.dumps(lock, indent=2))
        for seed in (2023, 42):
            run = args.out / "runs" / f"{trial_id}_seed{seed}"
            command = [str(PYTHON), str(LAUNCHER), "--split", str(args.split),
                       "--acceptance", str(args.acceptance), "--seed", str(seed),
                       "--shot", "5", "--methods", f"B0,{method}",
                       "--lock", str(lock_path), "--out", str(run)]
            commands.append(shlex.join(command))
            manifest.append(dict(trial_id=trial_id, method=method, seed=seed,
                                 config=config, run_dir=str(run),
                                 lock_path=str(lock_path)))
    (args.out / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    (args.out / "trial_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "grid.json").write_text(json.dumps(
        dict(budgets=[8, 16, 32], true_grid=true_grid,
             support_seeds=[2023, 42], n_trials=len(trials),
             role="development-only B1/B2/B3 selection"), indent=2))
    print(f"froze {len(trials)} candidates x two supports")


if __name__ == "__main__":
    main()
