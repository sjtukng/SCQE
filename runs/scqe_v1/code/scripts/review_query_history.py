"""Inventory recorded query outputs before attesting DIOR held-out history."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
OLD = ROOT / "runs/scqe_v1"


def read_ids(path):
    return {int(x["image_id"]) for x in json.loads(path.read_text())["queries"]}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(value, found):
    if isinstance(value, list):
        for child in value:
            collect(child, found)
    elif isinstance(value, dict):
        for key in ("image_id", "query_id"):
            if key in value:
                try:
                    found.add(int(value[key]))
                except (TypeError, ValueError):
                    pass
        # Historical proposal/detection JSON often uses query ID as the
        # top-level key and has boxes/scores in the value.
        for key, child in value.items():
            if isinstance(child, dict) and ("boxes" in child or "scores" in child):
                try:
                    found.add(int(key))
                except ValueError:
                    pass
            collect(child, found)


def main():
    splits = OLD / "splits"
    dev = read_ids(splits / "dior_dcal40.json") | read_ids(splits / "dior_dcheck60.json")
    final = read_ids(splits / "dior_final_queries.json")
    records = []
    for stage in ("e1_validation", "e2_calibration", "e3_pilot"):
        for path in sorted((OLD / stage).rglob("*.json")):
            found = set()
            collect(json.loads(path.read_text()), found)
            if found:
                records.append(dict(path=str(path), kind=stage,
                                    count=len(found), dev_count=len(found & dev),
                                    final_count=len(found & final),
                                    outside_dev_count=len(found - dev),
                                    final_ids=sorted(found & final), sha256=sha(path)))
    for path in sorted((OLD / "backend").glob(
            "seed*_shot*/adapt/inference/coco_instances_results.json")):
        found = set()
        collect(json.loads(path.read_text()), found)
        records.append(dict(path=str(path), kind="adaptation_dev_evaluation",
                            count=len(found), dev_count=len(found & dev),
                            final_count=len(found & final),
                            outside_dev_count=len(found - dev),
                            final_ids=sorted(found & final), sha256=sha(path)))
    for path in sorted((ROOT / "runs/sqr_v1").glob("p*/*per_query*.csv")):
        with path.open(newline="") as f:
            table = csv.DictReader(f)
            found = {int(row["query_id"]) for row in table if row.get("query_id")}
        records.append(dict(path=str(path), kind="sqr_historical_query_csv",
                            count=len(found), dev_count=len(found & dev),
                            final_count=len(found & final),
                            outside_dev_count=len(found - dev),
                            final_ids=sorted(found & final), sha256=sha(path)))
    by_kind = defaultdict(set)
    for rec in records:
        by_kind[rec["kind"]].add(rec["path"])
    summary = dict(status="WORKSPACE_REVIEW_ONLY",
                   dev_count=len(dev), final_count=len(final),
                   scanned_files=len(records),
                   scanned_by_kind={k: len(v) for k, v in by_kind.items()},
                   files_with_final_query_ids=[r for r in records if r["final_count"]],
                   files_with_outside_dev_ids=[r for r in records if r["outside_dev_count"]],
                   records=records,
                   limitation="A workspace artifact scan cannot establish use outside this server. "
                   "The user must confirm that history separately.")
    out = RUN / "audit/workspace_query_history_review.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"scanned {len(records)} query artifact files; "
          f"final overlaps: {len(summary['files_with_final_query_ids'])}; "
          f"outside-dev files: {len(summary['files_with_outside_dev_ids'])}")


if __name__ == "__main__":
    main()
