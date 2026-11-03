"""E8 fixed-20-image P/M sensitivity summary using locked D thresholds."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate

EXPECTED = {(4, 32), (16, 32), (32, 32), (16, 8), (16, 16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []
    seen = set()
    lock_hashes = set()
    split_hashes = set()
    for run_path in args.run:
        path = Path(run_path)
        launch = json.loads((path / "launch.json").read_text())
        meta = json.loads((path / "metadata.json").read_text())
        if launch["status"] != "COMPLETE" or meta["seed"] != 2023 or \
                meta["shot"] != 5 or meta["expected_queries"] != 20:
            raise RuntimeError(f"invalid E8 run: {path}")
        P = meta["probes"]
        M = meta["budget_override"] or 32
        seen.add((P, M))
        lock_hashes.add(meta["lock_sha256"])
        split_hashes.add(meta["split_sha256"])
        plans = [json.loads(p.read_text()) for p in sorted(
            (path / "plans").glob("*__SCQE.json"))]
        if len(plans) != 20:
            raise RuntimeError(f"missing SCQE plan records: {path}")
        method_ms = []
        for plan in plans:
            qid = plan["image_id"]
            worker = json.loads((path / "timing" / f"{qid}__SCQE.json").read_text())
            method_ms.append(plan["shared_preprocess_encoder_ms"] +
                             plan["shared_evidence_ms"] +
                             plan["method_gpu0_ms"] + worker["backend_ms"])
        metric = evaluate(json.loads((path / "predictions_SCQE.json").read_text()),
                          args.gt)
        rows.append(dict(P=P, M=M, AP=metric["AP"], AP50=metric["AP50"],
                         mean_component_sum_ms=statistics.mean(method_ms),
                         planned_exchanges=sum(x["planned_m"] for x in plans),
                         deployed_exchanges=sum(x["accepted_m"] for x in plans),
                         rejected=sum(x["status"] == "true_reject" for x in plans),
                         empty=sum(x["planned_m"] == 0 for x in plans),
                         run=str(path), cost_note="accuracy-run component sum; not fresh E9 wall latency"))
    if seen != EXPECTED or len(rows) != 5 or len(lock_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("need five distinct locked configurations on the same support")
    args.out.write_text(json.dumps(sorted(rows, key=lambda r: (r["P"], r["M"])), indent=2))


if __name__ == "__main__":
    main()
