"""E7 per-GT DIOR geometry trajectories and prespecified qualitative cases."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def xyxy_from_cxcywh(reference, state):
    W, H = state["img_W"], state["img_H"]
    sx, sy = state["scale_factor"]
    cx, cy, w, h = reference.unbind(-1)
    return torch.stack([(cx - w / 2) * W / sx, (cy - h / 2) * H / sy,
                        (cx + w / 2) * W / sx, (cy + h / 2) * H / sy], -1)


def all_iou(boxes, proposals):
    b = torch.as_tensor(boxes, dtype=torch.float64, device="cuda:0").reshape(-1, 4)
    p = torch.as_tensor(proposals, dtype=torch.float64, device="cuda:0").reshape(-1, 4)
    if not len(b) or not len(p):
        return torch.zeros((len(b), len(p)), dtype=torch.float64, device="cuda:0")
    low = torch.maximum(b[:, None, :2], p[None, :, :2])
    high = torch.minimum(b[:, None, 2:], p[None, :, 2:])
    intersection = (high - low).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)[:, None]
    area_p = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(-1)[None]
    return intersection / (area_b + area_p - intersection).clamp(min=1e-12)


def max_iou_all(boxes, proposals):
    matrix = all_iou(boxes, proposals)
    return matrix.max(-1).values.tolist() if matrix.shape[1] else [0.0] * matrix.shape[0]


def gt_box(ann):
    x, y, w, h = ann["bbox"]
    return [x, y, x + w, y + h]


def det_boxes(detections, min_score=0.0):
    return [[*d["bbox"][:2], d["bbox"][0] + d["bbox"][2],
             d["bbox"][1] + d["bbox"][3]]
            for d in detections if d["score"] >= min_score]


def tp_fp(detections, annotations):
    # DIOR conversion has ordinary, non-crowd GT. Match score order within
    # category at IoU .50, with each GT claimed at most once.
    claimed = set()
    gt_by_class = {}
    for ann in annotations:
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            continue
        gt_by_class.setdefault(int(ann["category_id"]), []).append(ann)
    tp = fp = 0
    for det in sorted(detections, key=lambda x: -x["score"]):
        box = det_boxes([det])[0]
        class_anns = [ann for ann in gt_by_class.get(int(det["category_id"]), [])
                      if int(ann["id"]) not in claimed]
        values = all_iou([box], [gt_box(ann) for ann in class_anns])[0].tolist()
        candidates = list(zip(values, (int(ann["id"]) for ann in class_anns)))
        if candidates:
            best_iou, ann_id = max(candidates, key=lambda x: (x[0], -x[1]))
            if best_iou >= 0.5:
                claimed.add(ann_id)
                tp += 1
                continue
        fp += 1
    return tp, fp, claimed


def median_case(rows, key):
    values = [r[key] for r in rows]
    center = statistics.median(values)
    return min(rows, key=lambda r: (abs(r[key] - center), str(r["image_id"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--display-threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; no CPU tensor fallback")
    run_meta = json.loads((args.run / "metadata.json").read_text())
    if json.loads((args.run / "launch.json").read_text())["status"] != "COMPLETE":
        raise RuntimeError("coverage diagnostic requires a complete paired run")
    if run_meta["lock_sha256"] != hashlib.sha256(args.lock.read_bytes()).hexdigest():
        raise ValueError("run and diagnostic lock differ")
    if run_meta["dataset"] != "DIOR":
        raise ValueError("this diagnostic only handles DIOR image parents")
    gt = json.loads(args.gt.read_text())
    gt_ids = {int(image["id"]) for image in gt["images"]}
    plan_paths = sorted((args.run / "plans").glob("*__SCQE.json"))
    plan_ids = {int(path.name.split("__")[0]) for path in plan_paths}
    if len(plan_paths) != run_meta["expected_queries"] or plan_ids != gt_ids:
        raise ValueError("SCQE plan images, run count and GT images must match exactly")
    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    lock = json.loads(args.lock.read_text())
    ep = lock["selections"]["SCQE"]["eps_proxy"]
    et = lock["selections"]["SCQE"]["eps_true"]
    query_rows, trajectories, cases = [], [], []
    for path in plan_paths:
        qid = int(path.name.split("__")[0])
        plan = json.loads(path.read_text())
        st = torch.load(args.run / "states" / f"{qid}.pt", map_location="cuda:0")
        base_job = json.loads((args.run / "jobs" / f"{qid}__B0.ready.json").read_text())
        deployed_job = json.loads((args.run / "jobs" / f"{qid}__SCQE.ready.json").read_text())
        planned_file = args.run / "planned_proposals" / f"{qid}__SCQE.json"
        planned_boxes = json.loads(planned_file.read_text())["boxes"] if planned_file.exists() \
            else base_job["boxes"]
        base_dets = json.loads((args.run / "detections" / f"{qid}__B0.json").read_text())
        deployed_dets = json.loads((args.run / "detections" / f"{qid}__SCQE.json").read_text())
        annotations = gt_by_image.get(qid, [])
        base_tp, base_fp, base_matched = tp_fp(base_dets, annotations)
        deployed_tp, deployed_fp, deployed_matched = tp_fp(deployed_dets, annotations)
        dq = (deployed_tp - base_tp) - (deployed_fp - base_fp)
        initial_refs = xyxy_from_cxcywh(st["xi_all"][st["A0"]], st).tolist()
        planned_refs = xyxy_from_cxcywh(
            st["xi_all"][torch.tensor(plan["Astar"], device="cuda:0")], st).tolist()
        deployed_refs = xyxy_from_cxcywh(
            st["xi_all"][torch.tensor(plan["Aout"], device="cuda:0")], st).tolist()
        base_det_boxes = det_boxes(base_dets)
        deployed_det_boxes = det_boxes(deployed_dets)
        boxes_gt = [gt_box(ann) for ann in annotations]
        stages = {
            "initial_reference_iou": max_iou_all(boxes_gt, initial_refs),
            "planned_reference_iou": max_iou_all(boxes_gt, planned_refs),
            "deployed_reference_iou": max_iou_all(boxes_gt, deployed_refs),
            "B0_proposal_iou": max_iou_all(boxes_gt, base_job["boxes"]),
            "planned_proposal_iou": max_iou_all(boxes_gt, planned_boxes),
            "deployed_proposal_iou": max_iou_all(boxes_gt, deployed_job["boxes"]),
            "B0_final_iou": max_iou_all(boxes_gt, base_det_boxes),
            "deployed_final_iou": max_iou_all(boxes_gt, deployed_det_boxes),
        }
        for ai, ann in enumerate(annotations):
            box = gt_box(ann)
            initial = stages["initial_reference_iou"][ai]
            planned_ref = stages["planned_reference_iou"][ai]
            deployed_ref = stages["deployed_reference_iou"][ai]
            base_prop = stages["B0_proposal_iou"][ai]
            planned_prop = stages["planned_proposal_iou"][ai]
            deployed_prop = stages["deployed_proposal_iou"][ai]
            base_final = stages["B0_final_iou"][ai]
            deployed_final = stages["deployed_final_iou"][ai]
            trajectories.append(dict(image_id=qid, support_seed=run_meta["seed"],
                                     gt_id=int(ann["id"]),
                                     category_id=int(ann["category_id"]),
                                     initial_reference_iou=initial,
                                     planned_reference_iou=planned_ref,
                                     deployed_reference_iou=deployed_ref,
                                     B0_proposal_iou=base_prop,
                                     planned_proposal_iou=planned_prop,
                                     deployed_proposal_iou=deployed_prop,
                                     B0_final_iou=base_final,
                                     deployed_final_iou=deployed_final,
                                     deployed_minus_B0=deployed_final - base_final,
                                     B0_classified_TP=int(ann["id"]) in base_matched,
                                     deployed_classified_TP=int(ann["id"]) in deployed_matched,
                                     gained50=base_final < .5 <= deployed_final,
                                     lost50=deployed_final < .5 <= base_final,
                                     gained75=base_final < .75 <= deployed_final,
                                     lost75=deployed_final < .75 <= base_final))
        query_rows.append(dict(image_id=qid, support_seed=run_meta["seed"],
                               image_path=plan["image_path"],
                               status=plan["status"], dq=dq,
                               B0_tp=base_tp, B0_fp=base_fp,
                               deployed_tp=deployed_tp, deployed_fp=deployed_fp,
                               proxy_risk=plan.get("proxy_cum"),
                               true_risk=plan.get("true_risk"),
                               rho_t=plan["true_risk"] / et if plan.get("true_risk") is not None else None,
                               planned_m=plan["planned_m"], accepted_m=plan["accepted_m"],
                               fallback_verified=plan["fallback_verified"]))
    positive = [r for r in query_rows if r["status"] == "accepted" and r["dq"] > 0]
    negative = [r for r in query_rows if r["status"] == "accepted" and r["dq"] < 0]
    rejected = [r for r in query_rows if r["status"] == "true_reject" and r["rho_t"] is not None]
    selections = []
    if run_meta["seed"] == 2023 and run_meta["shot"] == 5:
        for label, group, key in (("accepted benefit", positive, "dq"),
                                  ("accepted loss", negative, "dq"),
                                  ("rejected", rejected, "rho_t")):
            if group:
                chosen = dict(median_case(group, key), type=label)
                selections.append(chosen)
    for selected in selections:
        qid = selected["image_id"]
        plan = json.loads((args.run / "plans" / f"{qid}__SCQE.json").read_text())
        st = torch.load(args.run / "states" / f"{qid}.pt", map_location="cuda:0")
        slots = [s["slot"] for s in plan["steps"]]
        def ref_boxes(key):
            idx = torch.tensor(plan[key], device="cuda:0")
            return xyxy_from_cxcywh(st["xi_all"][idx[slots]], st).tolist()
        b0_dets = json.loads((args.run / "detections" / f"{qid}__B0.json").read_text())
        scqe_dets = json.loads((args.run / "detections" / f"{qid}__SCQE.json").read_text())
        cases.append(dict(type=selected["type"], query_id=qid,
                          image_path=selected["image_path"],
                          status=selected["status"], support_seed=run_meta["seed"],
                          planned_m=selected["planned_m"],
                          accepted_m=selected["accepted_m"],
                          proxy_risk=selected["proxy_risk"], true_risk=selected["true_risk"],
                          epsilon_proxy=ep, epsilon_true=et,
                          fallback_verified=selected["fallback_verified"],
                          slots=slots,
                          outgoing=[plan["A0"][slot] for slot in slots],
                          incoming=[plan["Astar"][slot] for slot in slots],
                          initial_refs=ref_boxes("A0"), planned_refs=ref_boxes("Astar"),
                          b0_detections=det_boxes(b0_dets, args.display_threshold),
                          deployed_detections=det_boxes(scqe_dets, args.display_threshold)))
    args.out.mkdir(parents=True)
    (args.out / "coverage_trajectories.json").write_text(json.dumps(trajectories, indent=2))
    (args.out / "query_tp_fp.json").write_text(json.dumps(query_rows, indent=2))
    (args.out / "qualitative_cases.json").write_text(json.dumps(cases, indent=2))
    (args.out / "missing_case_types.json").write_text(json.dumps(
        [name for name in ("accepted benefit", "accepted loss", "rejected")
         if name not in {c["type"] for c in cases}], indent=2))


if __name__ == "__main__":
    main()
