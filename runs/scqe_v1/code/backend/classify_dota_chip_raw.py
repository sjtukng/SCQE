"""Classify frozen DOTA chip proposal jobs without chip-level postprocessing.

One adapted CD-ViTO model runs on physical GPU 0 after PF-RPN has finished.
The gzip binary stream stores float32 score/box tensors for every proposal
and class, before any threshold or chip-level postprocessing. Parent-level
policy runs later. This bounds storage for the 5297-chip formal split.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
REPO = ROOT / "third_party/CDFSOD-benchmark"
sys.path.insert(0, str(REPO))
os.environ.setdefault("DETECTRON2_DATASETS", str(ROOT / "c0_datasets"))

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeTransform
from detectron2.modeling import build_model
import lib.data.fewshot, lib.data.ovdshot, lib.data.lvis  # noqa: F401


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(seed, shot, prototype, checkpoint):
    import lib.dinov2.layers.attention as attention
    attention.XFORMERS_AVAILABLE = False
    torch.use_deterministic_algorithms(True)
    cfg = get_cfg()
    cfg.DE.CONTROLLER = False
    cfg.merge_from_file(str(REPO / f"configs/dota_hbb/vitl_shot{shot}_dota_hbb_finetune.yaml"))
    cfg.merge_from_list([
        "DATASETS.TRAIN", f"('DOTA_HBB_{shot}shot_seed{seed}',)",
        "DATASETS.TEST", "()",
        "DE.CLASS_PROTOTYPES", str(prototype),
        "DE.OFFLINE_RPN_CONFIG", str(REPO / "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"),
        "DE.BG_PROTOTYPES", str(REPO / "weights/initial/background/background_prototypes.vitl14.pth"),
        "DE.EXTERNAL_PROPOSALS_FILE", "__scqe_inline__",
    ])
    cfg.freeze()
    if cfg.DE.MULTIPLY_RPN_SCORE or cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST != 0.001 or \
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST != 0.5 or \
            cfg.TEST.DETECTIONS_PER_IMAGE != 100 or cfg.DATASETS.TEST:
        raise RuntimeError("DOTA classifier/postprocessing contract differs from the frozen policy")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(checkpoint))
    model = model.to("cuda:0").eval()
    model._scqe_return_raw_detections = True
    return model


def classify(model, record):
    chip_id = record["chip_id"]
    file_name = Path(record["file_name"])
    if file_name.stem != chip_id or int(record["width"]) != 1024 or \
            int(record["height"]) != 1024:
        raise ValueError(f"chip job metadata mismatch: {chip_id}")
    image = cv2.imread(str(file_name), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (1024, 1024):
        raise RuntimeError(f"chip cannot be decoded: {file_name}")
    boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)
    if len(boxes) > 900 or len(boxes) != len(scores) or \
            not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError(f"invalid frozen PF proposals: {chip_id}")
    if not len(boxes):
        return (np.empty((0, 15), dtype="<f4"),
                np.empty((0, 15, 4), dtype="<f4"))
    # Match the frozen CD-ViTO 800-pixel test resize. The model's external
    # proposal hook expects coordinates in the actual tensor/image size.
    scale = 800.0 / 1024.0
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = ResizeTransform(1024, 1024, 800, 800).apply_image(rgb).copy()
    inp = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    model._external_proposals = {
        file_name.name: {"boxes": boxes * scale, "scores": scores}}
    with torch.no_grad():
        raw = model([dict(image=inp, height=800, width=800,
                          file_name=str(file_name))])[0]
    class_scores = raw["raw_scores"].detach().cpu().numpy()
    predicted = raw["raw_boxes"].detach().cpu().numpy()
    if class_scores.shape != (len(boxes), 15) or predicted.shape not in \
            ((len(boxes), 4), (len(boxes), 60)):
        raise RuntimeError(f"unexpected DOTA raw tensor shapes: {class_scores.shape}, {predicted.shape}")
    if not np.isfinite(class_scores).all() or not np.isfinite(predicted).all():
        raise RuntimeError(f"nonfinite DOTA classifier output: {chip_id}")
    predicted = predicted.reshape(len(boxes), -1, 4) / scale
    if predicted.shape[1] == 1:
        predicted = np.broadcast_to(predicted, (len(boxes), 15, 4))
    return (np.asarray(class_scores, dtype="<f4", order="C"),
            np.asarray(predicted, dtype="<f4", order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2023, 42, 1234, 7, 99), required=True)
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--method", choices=("B0", "B1", "B2", "B3", "SCQE"), required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.policy.exists():
        raise FileExistsError("raw DOTA classifier outputs are write-once")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("DOTA classification requires exactly one visible CUDA GPU")
    files = sorted(args.jobs.glob(f"*__{args.method}.ready.json"))
    if len(files) != 5297 or not (args.jobs.parent / "producer_done.json").is_file():
        raise RuntimeError("DOTA chip proposal stage is incomplete")
    backend = ROOT / f"runs/scqe_v1/backend_dota/seed{args.seed}_shot{args.shot}"
    prototype = backend / f"DOTA_HBB_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    checkpoint = backend / "adapt/model_final.pth"
    if not prototype.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"DOTA adaptation is incomplete: {backend}")
    build_seed = 20261014 + args.seed * 17 + args.shot
    random.seed(build_seed)
    np.random.seed(build_seed)
    torch.manual_seed(build_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(args.seed, args.shot, prototype, checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(args.out, "wb", compresslevel=3) as stream:
        stream.write(b"SCQEDOTA1")
        for index, path in enumerate(files, 1):
            record = json.loads(path.read_text())
            if record.get("method") != args.method:
                raise RuntimeError(f"method mismatch: {path}")
            scores, boxes = classify(model, record)
            name = record["chip_id"].encode("ascii")
            if len(name) > 65535 or len(scores) > 900:
                raise RuntimeError("DOTA binary frame limits exceeded")
            stream.write(struct.pack("<HH", len(name), len(scores)))
            stream.write(name)
            stream.write(scores.tobytes())
            stream.write(boxes.tobytes())
            count += scores.size
            if index % 100 == 0:
                print(f"[{index}/5297] {args.method} raw DOTA chips", flush=True)
    policy = dict(dataset="DOTA-HBB", model_class_count=15,
                  per_chip_score_threshold=None, per_chip_nms=False, per_chip_cap=None,
                  bbox_format="xyxy_float32_abs_original_chip", score_source="classifier_only",
                  raw_format="SCQEDOTA1_gzip_float32_frames",
                  physical_gpu=0, checkpoint_sha256=sha(checkpoint),
                  prototype_sha256=sha(prototype), raw_sha256=sha(args.out),
                  chips=len(files), raw_class_detections=count)
    args.policy.write_text(json.dumps(policy, indent=2) + "\n")
    print(f"wrote {count} unfiltered DOTA class scores and boxes: {args.out}")


if __name__ == "__main__":
    main()
