"""Combine five completed E7 per-support GT trajectories without recomputation."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=Path, required=True,
                    help="directory produced by diagnose_coverage.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if len(args.input) != 5:
        raise ValueError("five separate support diagnostic directories required")
    trajectories, query_rows = [], []
    seeds = set()
    gt_sets = []
    query_sets = []
    for directory in args.input:
        traj = json.loads((directory / "coverage_trajectories.json").read_text())
        queries = json.loads((directory / "query_tp_fp.json").read_text())
        these_seeds = {int(row["support_seed"]) for row in traj + queries}
        if len(these_seeds) != 1:
            raise ValueError(f"one support seed required per input: {directory}")
        seed = these_seeds.pop()
        if seed in seeds:
            raise ValueError(f"duplicate support seed: {seed}")
        seeds.add(seed)
        gt_keys = {(int(row["image_id"]), int(row["gt_id"])) for row in traj}
        query_ids = {int(row["image_id"]) for row in queries}
        if len(gt_keys) != len(traj) or len(query_ids) != len(queries):
            raise ValueError(f"duplicate GT/query identity: {directory}")
        gt_sets.append(gt_keys)
        query_sets.append(query_ids)
        trajectories.extend(traj)
        query_rows.extend(queries)
    if seeds != {2023, 42, 1234, 7, 99} or any(x != gt_sets[0] for x in gt_sets[1:]) or \
            any(x != query_sets[0] for x in query_sets[1:]):
        raise ValueError("five supports must cover identical GT and query identities")
    args.out.mkdir(parents=True)
    (args.out / "coverage_trajectories.json").write_text(json.dumps(trajectories, indent=2))
    (args.out / "query_tp_fp.json").write_text(json.dumps(query_rows, indent=2))
    (args.out / "manifest.json").write_text(json.dumps(
        dict(support_seeds=sorted(seeds), query_count=len(query_sets[0]),
             gt_per_support=len(gt_sets[0]), source_dirs=[str(x) for x in args.input]), indent=2))


if __name__ == "__main__":
    main()
