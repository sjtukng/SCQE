"""CD-ViTO worker consuming atomic proposals from the PF-RPN producer."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_proposals import ROOT, REPO, build, classify_one


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w") as f:
        json.dump(value, f, allow_nan=False)
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--baseline-from", type=Path)
    ap.add_argument("--same-job-cache", action="store_true")
    args = ap.parse_args()
    if args.device not in ("cuda:0", "cuda:1") or \
            not torch.cuda.is_available() or torch.cuda.device_count() <= int(args.device[-1]):
        raise RuntimeError("requested CUDA GPU is unavailable; CPU fallback is disabled")
    gpu_index = int(args.device[-1])
    backend = ROOT / "runs/scqe_v1/backend" / f"seed{args.seed}_shot{args.shot}"
    proto = backend / f"DIOR_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl"
    weights = backend / "adapt" / "model_final.pth"
    cfg = REPO / f"configs/dior/vitl_shot{args.shot}_dior_finetune.yaml"
    for path in (proto, weights, cfg):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Fix model initialization and any stochastic inference independently of
    # the arrival order of proposal jobs. The checkpoint and prototypes are
    # still shared by every method within the support episode.
    build_seed = 20261014 + args.seed * 17 + args.shot
    random.seed(build_seed)
    np.random.seed(build_seed)
    torch.manual_seed(build_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build(cfg, args.seed, args.shot, proto, weights).to(args.device).eval()
    source_meta = None
    if args.baseline_from is not None:
        source_launch = json.loads((args.baseline_from / "launch.json").read_text())
        source_done = json.loads((args.baseline_from / "worker_done.json").read_text())
        source_meta = json.loads((args.baseline_from / "metadata.json").read_text())
        if source_launch.get("status") != "COMPLETE" or \
                source_done.get("deterministic_algorithms") is not True or \
                source_done.get("attention") != "DINOv2 standard GPU attention" or \
                source_meta.get("expected_queries") != 40:
            raise RuntimeError("B0 reuse source lacks completed deterministic 40-image provenance")
    (args.spool / "detections").mkdir(exist_ok=True)
    (args.spool / "timing").mkdir(exist_ok=True)
    finished = set()
    all_detections = {}
    baseline_reuse_count = 0
    same_job_cache = {}
    cached_image_order = []
    same_job_reuse_count = 0
    same_job_verify_count = 0
    idle = 0
    while True:
        jobs = sorted((args.spool / "jobs").glob("*.ready.json"))
        new_jobs = [path for path in jobs if path.name not in finished]
        for path in new_jobs:
            record = json.loads(path.read_text())
            job_seed = (20261014 + args.seed * 100003 + args.shot * 1009 +
                        int(record["image_id"]) * 37 +
                        sum((i + 1) * ord(c) for i, c in enumerate(record["method"]))) % (2 ** 32)
            random.seed(job_seed)
            np.random.seed(job_seed)
            torch.manual_seed(job_seed)
            torch.cuda.synchronize(gpu_index)
            torch.cuda.reset_peak_memory_stats(gpu_index)
            start_ns = time.perf_counter_ns()
            cache_source = None
            same_job_source_method = None
            same_job_verified = False
            image_id = int(record["image_id"])
            job_key = hashlib.sha256(json.dumps(
                {key: record[key] for key in ("file_name", "height", "width", "boxes", "scores")},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if source_meta is not None and record["method"] == "B0":
                current_meta = json.loads((args.spool / "metadata.json").read_text())
                keys = ("seed", "shot", "split_sha256", "support_sha256",
                        "checkpoint_sha256", "backend_checkpoint_sha256",
                        "backend_prototype_sha256")
                if any(current_meta.get(key) != source_meta.get(key) for key in keys):
                    raise RuntimeError("B0 reuse source has different dataset/support/model hashes")
                source_job = args.baseline_from / "jobs" / path.name
                source_det = args.baseline_from / "detections" / path.name.replace(".ready", "")
                if not source_job.is_file() or not source_det.is_file() or \
                        json.loads(source_job.read_text()) != record:
                    raise RuntimeError(f"B0 job differs from reuse source: {path.name}")
                detections = json.loads(source_det.read_text())
                cache_source = str(source_det)
                baseline_reuse_count += 1
            elif args.same_job_cache and job_key in same_job_cache.get(image_id, {}):
                cached = same_job_cache[image_id][job_key]
                same_job_source_method = cached["method"]
                if same_job_verify_count < 3:
                    detections = classify_one(model, record["file_name"], int(record["height"]),
                                              int(record["width"]), record["boxes"],
                                              record["scores"], args.device)
                    if detections != cached["detections"]:
                        raise RuntimeError("identical image/proposal job changed classifier output")
                    same_job_verify_count += 1
                    same_job_verified = True
                else:
                    detections = [dict(det) for det in cached["detections"]]
                    same_job_reuse_count += 1
            else:
                detections = classify_one(model, record["file_name"], int(record["height"]),
                                          int(record["width"]), record["boxes"],
                                          record["scores"], args.device)
            if args.same_job_cache and image_id not in same_job_cache:
                same_job_cache[image_id] = {}
                cached_image_order.append(image_id)
                if len(cached_image_order) > 4:
                    same_job_cache.pop(cached_image_order.pop(0))
            if args.same_job_cache and job_key not in same_job_cache[image_id]:
                same_job_cache[image_id][job_key] = dict(
                    method=record["method"],
                    detections=[{key: value for key, value in det.items() if key != "image_id"}
                                for det in detections])
            torch.cuda.synchronize(gpu_index)
            end_ns = time.perf_counter_ns()
            if len(detections) > 100:
                raise RuntimeError("backend emitted >100 final detections per DIOR image")
            for det in detections:
                det["image_id"] = int(record["image_id"])
            atomic_json(args.spool / "detections" / path.name.replace(".ready", ""), detections)
            atomic_json(args.spool / "timing" / path.name.replace(".ready", ""),
                        dict(backend_start_ns=start_ns, backend_end_ns=end_ns,
                             backend_ms=(end_ns - start_ns) / 1e6,
                             inference_seed=job_seed,
                             baseline_reused_from=cache_source,
                             source_detection_sha256=sha(Path(cache_source)) if cache_source else None,
                             same_job_source_method=same_job_source_method,
                             same_job_cache_verified=same_job_verified,
                             device=args.device,
                             gpu_peak_allocated=torch.cuda.max_memory_allocated(gpu_index),
                             gpu_peak_reserved=torch.cuda.max_memory_reserved(gpu_index),
                             gpu0_peak_allocated=torch.cuda.max_memory_allocated(gpu_index)
                             if gpu_index == 0 else None,
                             gpu0_peak_reserved=torch.cuda.max_memory_reserved(gpu_index)
                             if gpu_index == 0 else None,
                             gpu1_peak_allocated=torch.cuda.max_memory_allocated(gpu_index)
                             if gpu_index == 1 else None,
                             gpu1_peak_reserved=torch.cuda.max_memory_reserved(gpu_index)
                             if gpu_index == 1 else None))
            all_detections.setdefault(record["method"], []).extend(detections)
            finished.add(path.name)
            action = ("reused B0" if cache_source else
                      "reused identical job" if same_job_source_method and not same_job_verified else
                      "verified identical job" if same_job_verified else "classified")
            print(f"{action} {path.name} on {args.device}",
                  flush=True)
        done = args.spool / "producer_done.json"
        if done.exists():
            meta = json.loads((args.spool / "metadata.json").read_text())
            expected = meta.get("expected_jobs", meta["expected_queries"] * len(meta["methods"]))
            if len(jobs) > expected:
                raise RuntimeError(f"expected at most {expected} jobs, got {len(jobs)}")
            # The producer may have written the completion marker after this
            # iteration's glob snapshot. Refresh before deciding the queue is
            # incomplete; all expected jobs must be consumed before sealing.
            if len(finished) < expected:
                time.sleep(0.2)
                continue
            if len(jobs) != expected:
                raise RuntimeError(f"expected {expected} jobs, got {len(jobs)}")
            if meta.get("mode") != "timing":
                for method in meta["methods"]:
                    ordered = sorted(all_detections.get(method, []),
                                     key=lambda d: (d["image_id"], -d["score"],
                                                    d["category_id"], d["bbox"]))
                    atomic_json(args.spool / f"predictions_{method}.json",
                                ordered)
            atomic_json(args.spool / "worker_done.json",
                        dict(status="COMPLETE", jobs=len(finished), device=args.device,
                             build_seed=build_seed, deterministic_algorithms=True,
                             attention="DINOv2 standard GPU attention",
                             baseline_reuse_count=baseline_reuse_count,
                             baseline_from=str(args.baseline_from) if args.baseline_from else None,
                             same_job_cache_enabled=args.same_job_cache,
                             same_job_reuse_count=same_job_reuse_count,
                             same_job_verify_count=same_job_verify_count,
                             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
            return
        if not new_jobs:
            time.sleep(0.5)
            idle += 1
            if idle % 120 == 0:
                print(f"waiting for producer; classified {len(finished)} jobs", flush=True)
        else:
            idle = 0


if __name__ == "__main__":
    main()
