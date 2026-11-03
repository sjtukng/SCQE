"""Generate D calibration trial locks and exact two-GPU commands; does not run them."""
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.acceptance_gate import require_scqe_acceptance

ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = Path(__file__).with_name("run_dior_pair.py")
PYTHON = Path("/home/root-user/.conda/envs/sqr/bin/python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--baseline-source-dir", type=Path,
                    help="directory with complete B1_b8_seed2023/42 runs for exact B0 reuse")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scales = json.loads(args.scales.read_text())
    if scales.get("protocol") != "D" or scales.get("dataset") != "DIOR":
        raise ValueError("expected measured DIOR D scales")
    require_scqe_acceptance(args.acceptance)
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    if len(entries) != 40:
        raise ValueError("D calibration selection requires all 40 frozen images")
    if args.baseline_source_dir is not None:
        for seed in (2023, 42):
            source = args.baseline_source_dir / f"B1_b8_seed{seed}"
            if json.loads((source / "launch.json").read_text()).get("status") != "COMPLETE":
                raise RuntimeError(f"incomplete B0 source: {source}")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "locks").mkdir()
    sp, si, st = (float(scales[k]) for k in ("s_p", "s_p_isolated", "s_t"))
    true_grid = sorted({min(st / 3, 1.0), min(st, 1.0), min(3 * st, 1.0)})
    trials = []
    for method in ("B1", "B2"):
        for budget in (8, 16, 32):
            trials.append((f"{method}_b{budget}", method, dict(budget=budget)))
    for ti, threshold in enumerate(true_grid):
        trials.append((f"B3_t{ti}", "B3", dict(eps_true=threshold)))
    for method, scale in (("SCQE", sp), ("A3-Cal", si)):
        proxy_grid = (scale / 3, scale, 3 * scale)
        for pi, proxy in enumerate(proxy_grid):
            for ti, threshold in enumerate(true_grid):
                trials.append((f"{method}_p{pi}_t{ti}", method,
                               dict(eps_proxy=proxy, eps_true=threshold)))
    commands = []
    manifest = []
    for trial_id, method, cfg in trials:
        lock = dict(protocol="D", dataset="DIOR", status="CALIBRATION_TRIAL",
                    delta_F=scales["delta_F"], selections={method: cfg},
                    measured_scales=str(args.scales))
        lock_path = args.out / "locks" / f"{trial_id}.json"
        lock_path.write_text(json.dumps(lock, indent=2))
        for seed in (2023, 42):
            run_dir = args.out / "runs" / f"{trial_id}_seed{seed}"
            cmd = [str(PYTHON), str(LAUNCHER), "--split", str(args.split),
                   "--acceptance", str(args.acceptance), "--seed", str(seed),
                   "--shot", "5", "--methods", f"B0,{method}", "--lock",
                   str(lock_path), "--out", str(run_dir), "--record-state"]
            if args.baseline_source_dir is not None:
                cmd += ["--baseline-from", str(args.baseline_source_dir / f"B1_b8_seed{seed}")]
            commands.append(shlex.join(cmd))
            manifest.append(dict(trial_id=trial_id, method=method, seed=seed,
                                 config=cfg, run_dir=str(run_dir),
                                 lock_path=str(lock_path)))
    (args.out / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" +
                                           "\n".join(commands) + "\n")
    (args.out / "trial_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "grid.json").write_text(json.dumps(
        dict(proxy_grid=[sp / 3, sp, 3 * sp],
             isolated_proxy_grid=[si / 3, si, 3 * si],
             true_grid=true_grid, budgets=[8, 16, 32], seeds=[2023, 42]), indent=2))
    print(f"prepared {len(trials)} trials x 2 supports; commands.sh was not executed")


if __name__ == "__main__":
    main()
