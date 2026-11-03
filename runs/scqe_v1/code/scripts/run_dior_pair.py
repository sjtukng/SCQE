"""Launch a DIOR episode on two GPUs, or sequentially on one GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.acceptance_gate import require_scqe_acceptance
from scqe.config import CHECKPOINT_PATH
from resume_dior import stage_prefix

ROOT = Path(__file__).resolve().parents[4]
PF_PYTHON = Path("/home/root-user/.conda/envs/sqr/bin/python")
BACKEND_PYTHON = Path("/home/root-user/.conda/envs/cdvito/bin/python")
PRODUCER = Path(__file__).with_name("infer_dior_stream.py")
TIMING_PRODUCER = Path(__file__).with_name("timing_dior_stream.py")
WORKER = ROOT / "runs/scqe_v1/code/backend/classify_stream_worker.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shot", type=int, choices=(1, 5, 10), required=True)
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--audit", type=Path,
                    help="data_audit.json, required for 4,900-image held-out evaluation")
    ap.add_argument("--methods", default="B0,B1,B2,B3,SCQE")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record-state", action="store_true")
    ap.add_argument("--baseline-from", type=Path,
                    help="completed same-support calibration run supplying exact B0 jobs/detections")
    ap.add_argument("--same-job-cache", action="store_true",
                    help="accuracy-only GPU classifier reuse for identical image/proposal jobs")
    ap.add_argument("--timing", action="store_true",
                    help="E9 fresh 50-query x 3-round timing; support seed 2023, shot 5")
    ap.add_argument("--probes", type=int, choices=(4, 16, 32), default=16)
    ap.add_argument("--budget-override", type=int, choices=(8, 16, 32))
    ap.add_argument("--single-gpu", action="store_true",
                    help="run proposal generation, then classification, on one physical GPU")
    ap.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0,
                    help="physical GPU for sequential single-GPU execution")
    ap.add_argument("--resume-from", type=Path,
                    help="copy the complete producer prefix of an interrupted single-GPU run")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("output directory must not exist; runs are immutable")
    required = {
        "PF Python environment": PF_PYTHON,
        "backend Python environment": BACKEND_PYTHON,
        "query split": args.split,
        "D configuration lock (run E3 calibration_select.py first)": args.lock,
        "E2 implementation acceptance (run acceptance.py first)": args.acceptance,
    }
    if args.audit is not None:
        required["E0/E1 input audit (run audit_inputs.py first)"] = args.audit
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment prerequisites:\n  " +
                                "\n  ".join(missing) +
                                "\nFollow README.md E0-E3 before launching the formal split.")
    methods = args.methods.split(",")
    if not methods or any(not method for method in methods) or len(methods) != len(set(methods)):
        raise ValueError("methods must be nonempty and unique")
    acceptance = json.loads(args.acceptance.read_text())
    split_info = json.loads(args.split.read_text())
    query_count = len(split_info.get("queries", [])) if isinstance(split_info, dict) else len(split_info)
    forward = acceptance.get("forward_checks", [])
    forward_ok = len(forward) >= 3 and all(row.get("pass_forward") for row in forward)
    if methods == ["B0"]:
        if not forward_ok:
            raise RuntimeError("B0 requires three passing native-forward checks")
        lock = json.loads(args.lock.read_text())
        if lock.get("status") != "LOCKED_B0_FIXED" or \
                lock.get("protocol") != "D" or lock.get("dataset") != "DIOR" or \
                lock.get("selections") != {"B0": {}}:
            raise RuntimeError("B0 requires the audited fixed native baseline lock")
    elif set(methods).issubset({"B0", "B1", "B2", "B3"}):
        checks = acceptance.get("solver_checks", [])
        if not forward_ok or not acceptance.get("synthetic_cache_test") or \
                len(checks) < 3 or not all(row.get("passed") for row in checks):
            raise RuntimeError("non-derivative selectors require passing forward/cache/solver checks")
        lock = json.loads(args.lock.read_text())
        if lock.get("status") not in ("CALIBRATION_TRIAL_NOGRAD", "LOCKED_NOGRAD",
                                      "CALIBRATION_TRIAL", "LOCKED") or \
                lock.get("protocol") != "D" or lock.get("dataset") != "DIOR" or \
                not (set(methods) - {"B0"}).issubset(lock.get("selections", {})):
            raise RuntimeError("B1/B2/B3 require their D no-gradient calibration lock")
        if lock["status"] in ("CALIBRATION_TRIAL", "LOCKED"):
            require_scqe_acceptance(args.acceptance)
        if query_count == 4900 and lock.get("status") not in ("LOCKED_NOGRAD", "LOCKED"):
            raise RuntimeError("formal non-derivative evaluation requires a completed lock")
    else:
        require_scqe_acceptance(args.acceptance)
        lock = json.loads(args.lock.read_text())
        if lock.get("status") not in ("CALIBRATION_TRIAL", "LOCKED") or \
                lock.get("protocol") != "D" or lock.get("dataset") != "DIOR" or \
                not (set(methods) - {"B0"}).issubset(lock.get("selections", {})):
            raise RuntimeError("SCQE requires a D calibration trial or completed D lock")
        if query_count == 4900 and lock.get("status") != "LOCKED":
            raise RuntimeError("formal SCQE evaluation requires a completed D lock")
    if args.timing and (args.seed != 2023 or args.shot != 5 or args.record_state):
        raise ValueError("timing requires seed 2023, 5-shot and fresh uncached inference")
    if args.timing and (args.probes != 16 or args.budget_override is not None):
        raise ValueError("E9 timing uses the locked P=16, M=32 configuration")
    if args.timing and args.single_gpu:
        raise ValueError("E9 two-GPU timing cannot use sequential single-GPU execution")
    if args.baseline_from is not None:
        if query_count != 40 or "B0" not in methods or args.timing or \
                not args.baseline_from.is_dir():
            raise ValueError("baseline reuse is restricted to 40-image Dcal accuracy runs")
    if args.same_job_cache and args.timing:
        raise ValueError("fresh E9 timing cannot use the accuracy-stage job cache")
    if args.same_job_cache and args.baseline_from is not None:
        raise ValueError("choose one audited accuracy cache mode per run")
    if args.resume_from is not None and (not args.single_gpu or args.timing or
                                         args.baseline_from is not None):
        raise ValueError("resume requires a fresh single-GPU accuracy run")
    if not args.single_gpu and args.physical_gpu != 0:
        raise ValueError("physical GPU selection requires --single-gpu")
    source_role = split_info.get("meta", {}) if isinstance(split_info, dict) else {}
    if "formal" in str(source_role.get("subset", "")) or \
            "formal" in str(source_role.get("source", "")) or \
            "final" in str(source_role.get("source", "")):
        if args.audit is None or not json.loads(args.audit.read_text()).get("formal_ready"):
            raise RuntimeError("held-out DIOR run requires a passing history/data audit")
    gpu_lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
    if len(gpu_lines) < (args.physical_gpu + 1 if args.single_gpu else 2):
        raise RuntimeError("requested CUDA GPU assignment is unavailable")
    args.out.mkdir(parents=True)
    resume_prefix = 0
    if args.resume_from is not None:
        support = ROOT / "runs/sqr_v1/splits" / \
            f"dior_support_{args.shot}shot_seed{args.seed}.json"
        backend = ROOT / "runs/scqe_v1/backend" / f"seed{args.seed}_shot{args.shot}"
        resume_prefix = stage_prefix(
            args.resume_from, args.out, args.split, args.seed, args.shot, methods,
            args.record_state, args.lock, support, CHECKPOINT_PATH,
            backend / "adapt/model_final.pth",
            backend / f"DIOR_{args.shot}shot_seed{args.seed}.vitl14.bbox.p{args.shot}.sk.pkl")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.physical_gpu)
               if args.single_gpu else "0,1",
               OMP_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    producer = [str(PF_PYTHON), str(TIMING_PRODUCER if args.timing else PRODUCER),
                "--split", str(args.split),
                "--seed", str(args.seed), "--shot", str(args.shot),
                "--methods", args.methods, "--lock", str(args.lock),
                "--spool", str(args.out)]
    if args.record_state:
        producer.append("--record-state")
    if resume_prefix:
        producer += ["--resume-prefix", str(resume_prefix)]
    if not args.timing:
        producer += ["--probes", str(args.probes)]
        producer += ["--backend-device", "cuda:0" if args.single_gpu else "cuda:1"]
        if args.budget_override is not None:
            producer += ["--budget-override", str(args.budget_override)]
    worker = [str(BACKEND_PYTHON), str(WORKER), "--seed", str(args.seed),
              "--shot", str(args.shot), "--spool", str(args.out),
              "--device", "cuda:0" if args.single_gpu else "cuda:1"]
    if args.baseline_from is not None:
        worker += ["--baseline-from", str(args.baseline_from)]
    if args.same_job_cache:
        worker.append("--same-job-cache")
    meta = dict(status="RUNNING", start=time.time(),
                gpu_lines=[gpu_lines[args.physical_gpu]] if args.single_gpu else gpu_lines[:2],
                physical_gpu=args.physical_gpu if args.single_gpu else None,
                execution_mode="sequential_single_gpu" if args.single_gpu else "concurrent_two_gpu",
                timing=args.timing, audit=str(args.audit) if args.audit else None,
                resume_from=str(args.resume_from) if args.resume_from else None,
                resume_prefix_queries=resume_prefix,
                producer=producer, worker=worker, out=str(args.out))
    (args.out / "launch.json").write_text(json.dumps(meta, indent=2))
    pp = wp = None
    try:
        with open(args.out / "producer.log", "w") as pout, open(args.out / "worker.log", "w") as wout:
            pp = subprocess.Popen(producer, cwd=ROOT, env=env, stdout=pout, stderr=subprocess.STDOUT)
            if args.single_gpu:
                p_code = pp.wait()
                if p_code == 0:
                    wp = subprocess.Popen(worker, cwd=ROOT, env=env, stdout=wout,
                                          stderr=subprocess.STDOUT)
                    w_code = wp.wait()
                else:
                    w_code = None
            else:
                wp = subprocess.Popen(worker, cwd=ROOT, env=env, stdout=wout,
                                      stderr=subprocess.STDOUT)
                while pp.poll() is None and wp.poll() is None:
                    time.sleep(2)
                if pp.poll() is not None and pp.returncode != 0 and wp.poll() is None:
                    wp.terminate()
                if wp.poll() is not None and wp.returncode != 0 and pp.poll() is None:
                    pp.terminate()
                p_code, w_code = pp.wait(), wp.wait()
        meta.update(end=time.time(), producer_exit=p_code, worker_exit=w_code,
                    status="COMPLETE" if p_code == w_code == 0 else "FAILED")
        (args.out / "launch.json").write_text(json.dumps(meta, indent=2))
        if meta["status"] != "COMPLETE":
            raise RuntimeError("pair run failed; inspect producer.log and worker.log")
    except BaseException:
        for proc in (pp, wp):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait()
        raise


if __name__ == "__main__":
    main()
