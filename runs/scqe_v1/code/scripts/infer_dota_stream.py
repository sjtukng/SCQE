"""DOTA-HBB protocol D: frozen support, one PF-RPN/SCQE pass per query chip.

Run on physical GPU 0 before the adapted CD-ViTO raw classifier. Validation
annotations are never read. DIOR's completed D parameter lock is inherited.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, SEED_GLOBAL
from scqe.pf_runtime import PFRuntime
from scqe.pipeline import QueryEngine
from scqe.support import SupportEncoder
from infer_dior_stream import sha, atomic_json, as_json, params


ROOT = Path(__file__).resolve().parents[4]
SPLITS = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(2023, 42, 1234, 7, 99), required=True)
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--methods", default="B0,B1,B2,B3,SCQE")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--record-state", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("DOTA PF-RPN requires exactly one visible CUDA GPU")
    if (args.spool / "metadata.json").exists() or (args.spool / "jobs").exists():
        raise FileExistsError("DOTA spool already contains a run")
    audit = json.loads((SPLITS / "data_audit.json").read_text())
    integrity = json.loads((SPLITS / "integrity_report.json").read_text())
    if audit.get("status") != "DATA_PREPARED_MODEL_BACKEND_PENDING" or \
            integrity.get("status") != "PASS":
        raise RuntimeError("DOTA query/support inputs have not passed integrity audit")
    if args.split.absolute() != (SPLITS / "val_chips_1024_gap200.json").absolute():
        raise RuntimeError("DOTA formal query chip manifest must be the frozen validation split")
    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "LOCKED" or lock.get("protocol") != "D" or \
            lock.get("dataset") != "DIOR":
        raise RuntimeError("DOTA protocol D must inherit the completed DIOR D lock")
    methods = args.methods.split(",")
    if methods != ["B0", "B1", "B2", "B3", "SCQE"]:
        raise ValueError("formal DOTA method set must be B0,B1,B2,B3,SCQE")
    support_path = SPLITS / f"dota_support_{args.shot}shot_seed{args.seed}.json"
    support = json.loads(support_path.read_text())["support"]
    if len(support) != 15 * args.shot:
        raise RuntimeError("DOTA support annotation budget mismatch")
    instances = [dict(class_id=int(item["class_id"]),
                      image_path=item["image_path"],
                      box_xyxy=[float(v) for v in item["box_xyxy"]])
                 for item in support]
    backend = ROOT / f"runs/scqe_v1/backend_dota/seed{args.seed}_shot{args.shot}"
    backend_checkpoint = backend / "adapt/model_final.pth"
    backend_prototype = backend / \
        f"DOTA_HBB_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    for path in (backend_checkpoint, backend_prototype, CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    entries = json.loads(args.split.read_text())["chips"]
    if len(entries) != 5297 or len({item["chip_id"] for item in entries}) != len(entries):
        raise RuntimeError("frozen DOTA validation chip manifest is incomplete")
    args.spool.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "plans", "states", "planned_proposals"):
        (args.spool / name).mkdir()
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = PFRuntime(model, device="cuda:0")
    p, t, valid = SupportEncoder(runtime).encode_episode(instances, 15)
    atomic_json(args.spool / "metadata.json", dict(
        protocol="D", dataset="DOTA-HBB", shot=args.shot, seed=args.seed,
        split_sha256=sha(args.split), support_sha256=sha(support_path),
        lock_sha256=sha(args.lock), checkpoint_sha256=sha(CHECKPOINT_PATH),
        backend_checkpoint_sha256=sha(backend_checkpoint),
        backend_prototype_sha256=sha(backend_prototype), methods=methods,
        expected_queries=len(entries), execution_mode="sequential_single_gpu",
        gpu_proposal="cuda:0", gpu_backend="cuda:0", probes=16,
        state_format="torch_save_gzip_level1" if args.record_state else None))
    delta = float(lock["delta_F"])
    selections = lock["selections"]
    for index, entry in enumerate(entries):
        chip_id = entry["chip_id"]
        parent_id = str(entry["parent_id"])
        image_path = entry["image_path"]
        torch.cuda.synchronize(0)
        start = time.perf_counter_ns()
        engine = QueryEngine(runtime, "DOTA-HBB", parent_id, chip_id, probes=16)
        engine.prepare(image_path)
        torch.cuda.synchronize(0)
        prep_ms = (time.perf_counter_ns() - start) / 1e6
        torch.cuda.synchronize(0)
        start = time.perf_counter_ns()
        alpha, cols = (None, []) if engine.bypass_reason else \
            engine.build_alpha(p, t, valid)[:2]
        torch.cuda.synchronize(0)
        evidence_ms = (time.perf_counter_ns() - start) / 1e6
        for method in methods:
            torch.cuda.synchronize(0)
            start = time.perf_counter_ns()
            result = engine.run_method(method, alpha, cols,
                                       **params(method, selections, delta))
            torch.cuda.synchronize(0)
            method_ms = (time.perf_counter_ns() - start) / 1e6
            record = as_json(result)
            record.update(chip_id=chip_id, parent_id=int(parent_id),
                          image_path=image_path, query_index=index,
                          shared_preprocess_encoder_ms=prep_ms,
                          shared_evidence_ms=evidence_ms,
                          method_gpu0_ms=method_ms)
            atomic_json(args.spool / "plans" / f"{chip_id}__{method}.json", record)
            if "planned_boxes" in result:
                atomic_json(args.spool / "planned_proposals" /
                            f"{chip_id}__{method}.json",
                            dict(boxes=result["planned_boxes"].tolist(),
                                 scores=result["planned_scores"].tolist()))
            atomic_json(args.spool / "jobs" / f"{chip_id}__{method}.ready.json",
                        dict(chip_id=chip_id, parent_id=int(parent_id),
                             file_name=image_path, height=1024, width=1024,
                             method=method, boxes=result["boxes"].tolist(),
                             scores=result["scores"].tolist()))
        if args.record_state:
            with gzip.open(args.spool / "states" / f"{chip_id}.pt.gz", "wb",
                           compresslevel=1) as stream:
                torch.save(dict(R0=engine.state.R0, A0=engine.state.A0,
                                H=engine.state.H_slots, y0=engine.state.y0,
                                Wsqrt=engine.state.wsqrt, xi_all=engine.state.xi_all,
                                vs_idx=engine.ctx.vs_idx if engine.ctx else None,
                                vs_boxes=engine.ctx.vs_boxes if engine.ctx else None,
                                alpha=alpha, g=engine.g, p=p, t=t, class_valid=valid),
                           stream)
        print(f"[{index + 1}/{len(entries)}] {chip_id}", flush=True)
    atomic_json(args.spool / "producer_done.json", dict(
        status="COMPLETE", count=len(entries),
        no_risk_search_cache_verified=QueryEngine._no_risk_cache_checks))


if __name__ == "__main__":
    main()
