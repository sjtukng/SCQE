"""E6 paired per-query TP/FP and A2 actual net gains from completed DIOR runs."""
import argparse
import json
from pathlib import Path

import torch

from diagnose_coverage import tp_fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; no CPU computation fallback")
    launch = json.loads((args.run / "launch.json").read_text())
    meta = json.loads((args.run / "metadata.json").read_text())
    if launch["status"] != "COMPLETE" or meta["dataset"] != "DIOR":
        raise RuntimeError("completed DIOR ablation run required")
    methods = meta["methods"]
    if not {"B0", "SCQE", "A1", "A2"}.issubset(methods):
        raise ValueError("B0, SCQE, A1 and A2 records required")
    gt = json.loads(args.gt.read_text())
    ids = {int(x["id"]) for x in gt["images"]}
    if len(ids) != meta["expected_queries"]:
        raise ValueError("GT/run image count mismatch")
    annotations = {}
    for ann in gt["annotations"]:
        annotations.setdefault(int(ann["image_id"]), []).append(ann)
    rows = []
    for qid in sorted(ids):
        baseline = None
        for method in ["B0", *(m for m in methods if m != "B0")]:
            plan = json.loads((args.run / "plans" / f"{qid}__{method}.json").read_text())
            detections = json.loads((args.run / "detections" / f"{qid}__{method}.json").read_text())
            tp, fp, _ = tp_fp(detections, annotations.get(qid, []))
            if method == "B0":
                baseline = (tp, fp)
            steps = plan["steps"]
            rows.append(dict(image_id=qid, support_seed=meta["seed"], method=method,
                             status=plan["status"], TP50=tp, FP50=fp,
                             delta_TP50=tp - baseline[0] if baseline else None,
                             delta_FP50=fp - baseline[1] if baseline else None,
                             planned_m=plan["planned_m"], accepted_m=plan["accepted_m"],
                             actual_net_gain=sum(step["gain"] for step in steps),
                             negative_net_steps=sum(step["gain"] < 0 for step in steps),
                             add_only_score_sum=sum(step.get("add_gain", 0) for step in steps),
                             removal_loss_sum=sum(step.get("removal_loss", 0) for step in steps),
                             no_context_diagnostics=plan.get("no_context_diagnostics")
                             if method == "A1" else None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
