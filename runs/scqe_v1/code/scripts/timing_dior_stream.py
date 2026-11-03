"""E9 fresh, serial per-query timing on GPU 0 + concurrent GPU 1 worker."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR, SEED_GLOBAL
from scqe.data import support_instances
from scqe.pf_runtime import PFRuntime
from scqe.pipeline import QueryEngine
from scqe.support import SupportEncoder
from infer_dior_stream import atomic_json, params, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--methods", default="B0,B1,B2,B3,SCQE,A4")
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--spool", type=Path, required=True)
    args = ap.parse_args()
    if args.seed != 2023 or args.shot != 5:
        raise ValueError("E9 frozen DIOR support is seed 2023, 5-shot")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; CPU fallback disabled")
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    if len(entries) != 50:
        raise ValueError("E9 timing needs exactly 50 frozen query images")
    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "LOCKED":
        raise ValueError("E9 requires a completed locked configuration")
    if (args.spool / "jobs").exists():
        raise FileExistsError("timing spool already contains jobs")
    for name in ("jobs", "plans", "timing"):
        (args.spool / name).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    runtime = PFRuntime(model, "cuda:0")
    torch.cuda.synchronize(0)
    cold_start = time.perf_counter_ns()
    p, t, valid = SupportEncoder(runtime).encode_episode(
        support_instances(args.seed, args.shot), 20)
    torch.cuda.synchronize(0)
    cold_start_ms = (time.perf_counter_ns() - cold_start) / 1e6
    methods = args.methods.split(",")
    if len(set(methods)) != len(methods):
        raise ValueError("duplicate methods")
    expected_jobs = 5 * len(methods) + 3 * len(entries) * len(methods)
    metadata = dict(mode="timing", protocol="D", dataset="DIOR", shot=5, seed=2023,
                    split_sha256=sha(args.split), lock_sha256=sha(args.lock),
                    expected_queries=50, expected_jobs=expected_jobs,
                    methods=methods, warmup_per_method=5, rounds=3,
                    cold_start_ms=cold_start_ms)
    atomic_json(args.spool / "metadata.json", metadata)
    records = []
    for round_index in range(-1, 3):
        run_entries = entries[:5] if round_index < 0 else entries
        for qi, entry in enumerate(run_entries):
            ordered = methods[(qi + max(round_index, 0)) % len(methods):] + \
                      methods[:(qi + max(round_index, 0)) % len(methods)]
            for method in ordered:
                qid = str(entry["image_id"])
                job_id = f"{round_index + 1}_{qi:03d}_{qid}_{method}"
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                start_ns = time.perf_counter_ns()
                engine = QueryEngine(runtime, "DIOR", qid, search_cache=False).prepare(
                    str(DIOR_DIR / "images" / entry["image_path"]))
                alpha, cols = None, []
                if method != "B0":
                    alpha, cols, _ = engine.build_alpha(p, t, valid,
                                                        no_context=method == "A1")
                result = engine.run_method(method, alpha, cols,
                                           **params(method, lock["selections"], lock["delta_F"]))
                torch.cuda.synchronize(0)
                pf_end_ns = time.perf_counter_ns()
                gpu0_alloc = torch.cuda.max_memory_allocated(0)
                gpu0_reserved = torch.cuda.max_memory_reserved(0)
                atomic_json(args.spool / "plans" / f"{job_id}.json",
                            dict(method=method, image_id=int(qid), round=round_index,
                                 status=result["status"], planned_m=result["planned_m"],
                                 accepted_m=result["accepted_m"],
                                 true_risk=result["true_risk"]))
                atomic_json(args.spool / "jobs" / f"{job_id}.ready.json",
                            dict(image_id=int(qid), file_name=str(DIOR_DIR / "images" / entry["image_path"]),
                                 height=int(entry["height"]), width=int(entry["width"]),
                                 method=method, boxes=result["boxes"].tolist(),
                                 scores=result["scores"].tolist()))
                ack = args.spool / "timing" / f"{job_id}.json"
                while not ack.exists():
                    time.sleep(0.01)
                backend = json.loads(ack.read_text())
                torch.cuda.synchronize(0)
                end_ns = time.perf_counter_ns()
                calls = 1 + (len(result.get("prefix_info", [])) if method == "B3" else
                             int(result["planned_m"] > 0))
                records.append(dict(job_id=job_id, image_id=int(qid), method=method,
                                    round=round_index, warmup=round_index < 0,
                                    wall_ms=(end_ns - start_ns) / 1e6,
                                    pf_ms=(pf_end_ns - start_ns) / 1e6,
                                    backend_ms=backend["backend_ms"],
                                    transfer_wait_ms=(end_ns - pf_end_ns) / 1e6,
                                    gpu0_peak_allocated=gpu0_alloc,
                                    gpu0_peak_reserved=gpu0_reserved,
                                    gpu1_peak_allocated=backend["gpu1_peak_allocated"],
                                    gpu1_peak_reserved=backend["gpu1_peak_reserved"],
                                    vjp_calls=engine.probes if engine.g is not None else 0,
                                    native_decoder_calls=calls,
                                    classifier_calls=1,
                                    planned_m=result["planned_m"],
                                    accepted_m=result["accepted_m"],
                                    status=result["status"]))
                print(f"{job_id} {records[-1]['wall_ms']:.1f} ms", flush=True)
    atomic_json(args.spool / "timing_raw.json", records)
    summary = {}
    for method in methods:
        per_query = {}
        for rec in records:
            if rec["method"] == method and not rec["warmup"]:
                per_query.setdefault(rec["image_id"], []).append(rec["wall_ms"])
        if len(per_query) != 50 or any(len(x) != 3 for x in per_query.values()):
            raise RuntimeError(f"incomplete timing repetitions for {method}")
        means = [statistics.mean(x) for x in per_query.values()]
        sorted_means = sorted(means)
        method_records = [r for r in records if r["method"] == method and not r["warmup"]]
        summary[method] = dict(n_queries=50, repeats=3, raw_records=150,
                               mean_ms=statistics.mean(means),
                               median_ms=statistics.median(means),
                               p95_ms=float(np.percentile(sorted_means, 95)),
                               gpu0_max_allocated=max(r["gpu0_peak_allocated"] for r in method_records),
                               gpu1_max_allocated=max(r["gpu1_peak_allocated"] for r in method_records),
                               max_single_gpu_allocated=max(max(r["gpu0_peak_allocated"],
                                                                 r["gpu1_peak_allocated"])
                                                            for r in method_records))
    atomic_json(args.spool / "timing_summary.json",
                dict(cold_start_ms=cold_start_ms, summary=summary))
    atomic_json(args.spool / "producer_done.json",
                dict(status="COMPLETE", count=expected_jobs))


if __name__ == "__main__":
    main()
