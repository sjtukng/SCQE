"""CD-ViTO backend: classify PF-RPN proposals for arbitrary box sets.

Runs in the `cdvito` conda env. Loads an adapted checkpoint, injects external
proposals (boxes in ORIGINAL image coords) and writes COCO-format detections.

Usage:
  python classify_proposals.py --seed 2023 --shot 5 \
      --proposals /path/proposals.json --out /path/detections.json

proposals.json: {"<image_id>": {"file_name": str, "boxes": [[x1,y1,x2,y2],...],
                               "scores": [...]}}
detections.json: COCO detections list (bbox xywh, category_id 1..20, score).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = Path("/home/root-user/projects/PF_RPN_SR/third_party/CDFSOD-benchmark")
ROOT = Path("/home/root-user/projects/PF_RPN_SR")
sys.path.insert(0, str(REPO))
os.environ.setdefault("DETECTRON2_DATASETS", str(ROOT / "c0_datasets"))

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import lib.data.fewshot, lib.data.ovdshot, lib.data.lvis  # noqa: F401  (registers DIOR splits)


def build(cfg_file, seed, shot, proto, weights):
    # xFormers CUTLASS attention is nondeterministic on this CUDA stack.
    # DINOv2 supplies an equivalent standard attention path on the GPU.
    import lib.dinov2.layers.attention as dinov2_attention
    dinov2_attention.XFORMERS_AVAILABLE = False
    torch.use_deterministic_algorithms(True)
    cfg = get_cfg()
    cfg.DE.CONTROLLER = False  # matches tools/train_net.py default (no --controller)
    cfg.merge_from_file(str(cfg_file))
    cfg.merge_from_list([
        "DATASETS.TRAIN", f"('DIOR_{shot}shot_seed{seed}',)",
        "DATASETS.TEST", "('DIOR_dev100',)",
        "DE.CLASS_PROTOTYPES", str(proto),
        "DE.OFFLINE_RPN_CONFIG", str(REPO / "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"),
        "DE.BG_PROTOTYPES", str(REPO / "weights/initial/background/background_prototypes.vitl14.pth"),
        "DE.EXTERNAL_PROPOSALS_FILE", "__scqe_inline__",  # never read; we inject
    ])
    cfg.freeze()
    if cfg.DE.MULTIPLY_RPN_SCORE or cfg.TEST.DETECTIONS_PER_IMAGE != 100:
        raise RuntimeError("backend must keep target score independent of PF score and cap at 100")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(weights))
    model.eval()
    return model


def classify_one(model, file_name, H, W, boxes, scores, device):
    import cv2
    img = cv2.imread(file_name)
    assert img is not None, file_name
    img = img[:, :, ::-1].copy()  # BGR->RGB
    th, tw = img.shape[:2]
    # DIOR images are 800x800 -> mapper resize is identity; assert to be safe
    assert (th, tw) == (H, W), f"size mismatch {(th,tw)} vs {(H,W)}"
    inp = torch.from_numpy(img.copy()).permute(2, 0, 1).contiguous()
    batched_inputs = [dict(image=inp, height=H, width=W, file_name=file_name)]
    model._external_proposals = {
        os.path.basename(file_name): {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
        }}
    with torch.no_grad():
        out = model(batched_inputs)
    if isinstance(out, (list, tuple)):
        out = out[0]
    inst = out["instances"]
    dets = []
    bb = inst.pred_boxes.tensor.cpu().numpy()
    ss = inst.scores.cpu().numpy()
    cc = inst.pred_classes.cpu().numpy()
    for k in range(len(ss)):
        x1, y1, x2, y2 = bb[k]
        dets.append(dict(bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         score=float(ss[k]), category_id=int(cc[k]) + 1))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--images-json", default=str(
        ROOT / "runs/scqe_v1/splits/dior_dev_queries_src.json"),
        help="json with image_id -> {file_name, height, width}")
    args = ap.parse_args()

    backend = ROOT / "runs/scqe_v1" / "backend" / f"seed{args.seed}_shot{args.shot}"
    proto = backend / f"DIOR_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    weights = backend / "adapt" / "model_final.pth"
    model = build(REPO / f"configs/dior/vitl_shot{args.shot}_dior_finetune.yaml",
                  args.seed, args.shot, proto, weights)
    model.to(args.device)

    proposals = json.load(open(args.proposals))
    meta = json.load(open(args.images_json))
    out_all = []
    t_all = []
    for i, (iid, rec) in enumerate(proposals.items()):
        m = meta[iid]
        t0 = time.time()
        dets = classify_one(model, m["file_name"], int(m["height"]),
                            int(m["width"]), rec["boxes"], rec["scores"],
                            args.device)
        for d in dets:
            d["image_id"] = int(iid)
        out_all.extend(dets)
        t_all.append(time.time() - t0)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(proposals)}] last={t_all[-1]:.2f}s "
                  f"mean={np.mean(t_all):.2f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_all, open(args.out, "w"))
    print(f"wrote {len(out_all)} detections -> {args.out}; "
          f"mean cls time {np.mean(t_all):.3f}s")


if __name__ == "__main__":
    main()
