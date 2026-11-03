"""E0/E1 read-only input audit and frozen DIOR held-out COCO GT extraction."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPLITS = ROOT / "runs/scqe_v1/splits"
SUPPORTS = ROOT / "runs/sqr_v1/splits"
DATA = ROOT / "data/DIOR"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids(manifest):
    return {int(x["image_id"]) for x in read(manifest)["queries"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--history-attestation", type=Path)
    ap.add_argument("--gpu-evidence", type=Path,
                    help="previous nvidia-smi CSV from a verified two-GPU run")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    train = read(DATA / "annotations/train.json")
    test = read(DATA / "annotations/test.json")
    train_images = {int(x["id"]) for x in train["images"]}
    test_images = {int(x["id"]) for x in test["images"]}
    train_anns = {int(x["id"]): x for x in train["annotations"]}
    dcal = ids(SPLITS / "dior_dcal40.json")
    dcheck = ids(SPLITS / "dior_dcheck60.json")
    final = ids(SPLITS / "dior_final_queries.json")
    errors = []
    if len(dcal) != 40 or len(dcheck) != 60 or len(final) != 4900:
        errors.append("DIOR 40/60/4900 manifest counts differ from contract")
    if dcal & dcheck or dcal & final or dcheck & final:
        errors.append("query roles overlap")
    if dcal | dcheck | final != test_images:
        errors.append("query roles do not cover 5,000 converted DIOR test images")
    support_summary = {}
    for seed in (2023, 42, 1234, 7, 99):
        ann_sets = {}
        for shot in (1, 5, 10):
            path = SUPPORTS / f"dior_support_{shot}shot_seed{seed}.json"
            if not path.exists():
                errors.append(f"missing support manifest: {path}")
                continue
            items = read(path)["support"]
            counts = Counter(int(x["class_id"]) for x in items)
            if any(counts.get(c, 0) != shot for c in range(20)):
                errors.append(f"seed {seed} shot {shot}: class counts invalid")
            selected = set()
            for it in items:
                iid, image_id = int(it["instance_id"]), int(it["image_id"])
                ann = train_anns.get(iid)
                if ann is None or int(ann["image_id"]) != image_id or \
                        int(ann["category_id"]) != int(it["class_id"]) + 1:
                    errors.append(f"support annotation mismatch: {seed}/{shot}/{iid}")
                if image_id not in train_images:
                    errors.append(f"support image absent from train: {image_id}")
                selected.add(iid)
            if len(selected) != 20 * shot:
                errors.append(f"support instance IDs repeat: {seed}/{shot}")
            ann_sets[shot] = selected
            support_summary[f"{seed}/{shot}"] = dict(count=len(items), sha256=sha(path))
        if len(ann_sets) == 3 and not (ann_sets[1] <= ann_sets[5] <= ann_sets[10]):
            errors.append(f"support nesting invalid for seed {seed}")
    history = "UNVERIFIED"
    if args.history_attestation:
        doc = read(args.history_attestation)
        prior = {int(x) for x in doc["previously_exposed_query_ids"]}
        if not dcal | dcheck <= prior:
            errors.append("history attestation omits known development queries")
        if prior & final:
            errors.append("held-out manifest intersects previously exposed queries")
        if not doc.get("reviewed_sources") or not doc.get("reviewer"):
            errors.append("history attestation needs reviewed_sources and reviewer")
        if not errors:
            history = "ATTESTED"
    else:
        errors.append("4900-query development/use history has no attestation")
    if args.gpu_evidence:
        with args.gpu_evidence.open(newline="") as f:
            snapshot = list(csv.DictReader(f, skipinitialspace=True))
        gpu = [f"GPU {row['index']}: {row['name']} ({row['uuid']})"
               for row in snapshot]
        gpu_source = str(args.gpu_evidence)
        gpu_evidence_sha256 = sha(args.gpu_evidence)
    else:
        gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
        gpu_source = "live nvidia-smi -L"
        gpu_evidence_sha256 = None
    if len(gpu) < 2:
        errors.append("fewer than two CUDA GPUs visible")
    resources = dict(gpu=gpu, gpu_source=gpu_source,
                     gpu_evidence_sha256=gpu_evidence_sha256,
                     pf_checkpoint=str(ROOT / "checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth"),
                     pf_checkpoint_sha256=sha(ROOT / "checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth"),
                     train_annotation_sha256=sha(DATA / "annotations/train.json"),
                     test_annotation_sha256=sha(DATA / "annotations/test.json"),
                     support_manifests=support_summary)
    audit = dict(status="READY" if not errors else "BLOCKED",
                 formal_ready=not errors, history=history, errors=errors,
                 dcal_count=len(dcal), dcheck_count=len(dcheck), final_count=len(final),
                 query_ids_disjoint=not bool(dcal & dcheck or dcal & final or dcheck & final),
                 support_seeds=[2023, 42, 1234, 7, 99])
    (args.out / "data_audit.json").write_text(json.dumps(audit, indent=2))
    (args.out / "resources.json").write_text(json.dumps(resources, indent=2))
    # Keep empty-GT images; no predictions or calibration inspect this GT.
    gt = dict(info=test.get("info", {}), licenses=test.get("licenses", []),
              images=[x for x in test["images"] if int(x["id"]) in final],
              annotations=[x for x in test["annotations"] if int(x["image_id"]) in final],
              categories=test["categories"])
    (args.out / "dior_final_gt.json").write_text(json.dumps(gt))
    missing = ["DOTA-v1.0 parents/chips and adapted backend unavailable in this workspace",
               "Protocol K non-target calibration source and target-free parameter lock unavailable",
               "Original FT-FSOD/SI-ViTO resources unavailable"] + errors
    (args.out / "missing_inputs.md").write_text(
        "# Missing inputs and blocked stages\n\n" + "\n".join(f"- {line}" for line in missing) + "\n")
    entrypoints = dict(audit_inputs=str(Path(__file__).resolve()),
                       acceptance=str(Path(__file__).with_name("acceptance.py")),
                       calibration_scales=str(Path(__file__).with_name("calibration_scales.py")),
                       calibration_grid=str(Path(__file__).with_name("calibration_grid.py")),
                       calibration_select=str(Path(__file__).with_name("calibration_select.py")),
                       calibration_confirmation=str(Path(__file__).with_name("confirm_calibration.py")),
                       inference=str(Path(__file__).with_name("run_dior_pair.py")),
                       evaluation=str(Path(__file__).with_name("evaluate_runs.py")),
                       timing=str(Path(__file__).with_name("timing_dior_stream.py")),
                       plotting=str(Path(__file__).with_name("plot_results.py")),
                       reporting=str(Path(__file__).with_name("build_report.py")))
    (args.out / "entrypoints.json").write_text(json.dumps(entrypoints, indent=2))
    if errors:
        print("audit recorded blockers; inspect data_audit.json")
    else:
        print("DIOR D input audit ready")


if __name__ == "__main__":
    main()
