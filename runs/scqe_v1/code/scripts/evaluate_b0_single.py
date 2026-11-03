"""Evaluate one complete formal DIOR B0 support episode before all five finish."""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from offline_eval import evaluate


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.audit.read_text())
    launch = json.loads((args.run / "launch.json").read_text())
    meta = json.loads((args.run / "metadata.json").read_text())
    done = json.loads((args.run / "worker_done.json").read_text())
    if not audit.get("formal_ready") or launch["status"] != "COMPLETE" or \
            done.get("jobs") != 4900:
        raise RuntimeError("formal audit or episode completion check failed")
    if (meta["protocol"], meta["dataset"], meta["methods"],
            meta["expected_queries"]) != ("D", "DIOR", ["B0"], 4900) or \
            meta["shot"] not in (1, 5, 10):
        raise ValueError("run is not a formal DIOR D B0 episode")
    seed, shot = meta["seed"], meta["shot"]
    support = ROOT / f"runs/sqr_v1/splits/dior_support_{shot}shot_seed{seed}.json"
    backend = ROOT / f"runs/scqe_v1/backend/seed{seed}_shot{shot}"
    expected = {
        "split_sha256": ROOT / "runs/scqe_v1/splits/dior_final_queries.json",
        "support_sha256": support,
        "lock_sha256": ROOT / "runs/scqe_server_20261014/config_lock_B0_D.json",
        "backend_checkpoint_sha256": backend / "adapt/model_final.pth",
        "backend_prototype_sha256": backend / f"DIOR_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl",
    }
    for field, path in expected.items():
        if digest(path) != meta[field]:
            raise RuntimeError(f"{field} differs from the frozen episode")
    for dirname, glob in (("jobs", "*.ready.json"), ("plans", "*__B0.json"),
                          ("detections", "*__B0.json")):
        if len(list((args.run / dirname).glob(glob))) != 4900:
            raise RuntimeError(f"incomplete {dirname}")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.gt))
    gt_ids = set(gt.getImgIds())
    if len(gt_ids) != 4900:
        raise RuntimeError("formal GT image count differs from 4,900")
    pred_path = args.run / "predictions_B0.json"
    preds = json.loads(pred_path.read_text())
    if not {int(x["image_id"]) for x in preds}.issubset(gt_ids):
        raise RuntimeError("prediction contains non-held-out image")
    with contextlib.redirect_stdout(io.StringIO()):
        metric = evaluate(preds, gt)
    result = dict(status="COMPLETE", protocol="D", dataset="DIOR", shot=shot,
                  method="B0", support_seed=seed, run_id=args.run.name,
                  gt_path=str(args.gt), gt_sha256=digest(args.gt),
                  audit_path=str(args.audit), prediction_path=str(pred_path),
                  prediction_sha256=digest(pred_path),
                  split_hash=meta["split_sha256"], support_hash=meta["support_sha256"],
                  config_hash=meta["lock_sha256"],
                  checkpoint_hash=meta["checkpoint_sha256"],
                  backend_checkpoint_hash=meta["backend_checkpoint_sha256"],
                  backend_prototype_hash=meta["backend_prototype_sha256"],
                  **metric)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"seed {seed}: AP {metric['AP']:.6f}, AP50 {metric['AP50']:.6f}, "
          f"AP75 {metric['AP75']:.6f}")


if __name__ == "__main__":
    main()
