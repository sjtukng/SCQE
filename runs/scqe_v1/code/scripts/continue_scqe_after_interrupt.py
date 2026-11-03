"""Wait for an unused visible GPU, then continue frozen D protocol jobs."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
SCRIPTS = Path(__file__).resolve().parent
STATE = RUN / "continuation_status.json"
LOG = RUN / "continuation.log"
PF_PY = "/home/root-user/.conda/envs/sqr/bin/python"


def update(**values):
    data = json.loads(STATE.read_text()) if STATE.is_file() else {}
    data.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, STATE)


def gpu_apps(physical_gpu):
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, capture_output=True, timeout=20)
    if devices.returncode:
        raise RuntimeError(f"nvidia-smi device query failed: {devices.stderr.strip()}")
    uuids = {}
    for row in devices.stdout.splitlines():
        index, uuid = (part.strip() for part in row.split(",", 1))
        uuids[int(index)] = uuid
    if physical_gpu not in uuids:
        raise RuntimeError(f"physical GPU {physical_gpu} is unavailable")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return [pid.strip() for row in result.stdout.splitlines()
            for pid, uuid in [row.split(",", 1)] if uuid.strip() == uuids[physical_gpu]]


def other_gpu_queue_alive():
    needles = (b"/home/root-user/projects/TECR/scripts/run_continuation_service.py",
               b"/home/root-user/projects/TECR/scripts/run_full_matrix.py")
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if any(needle in cmd for needle in needles):
            return True
    return False


def run(command, stage):
    update(state="RUNNING", stage=stage, command=list(map(str, command)))
    with LOG.open("a") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {stage}: " +
                     " ".join(map(str, command)) + "\n")
        stream.flush()
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    lockfile = (RUN / "continuation.lock").open("w")
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("an SCQE continuation is already running")
    if args.check_only:
        print(json.dumps({"physical_gpu": args.physical_gpu,
                          "gpu_apps": gpu_apps(args.physical_gpu),
                          "other_gpu_queue_alive": other_gpu_queue_alive()}))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["SCQE_PHYSICAL_GPU"] = str(args.physical_gpu)
    idle_checks = 0
    while idle_checks < 3:
        apps = gpu_apps(args.physical_gpu)
        other_queue = other_gpu_queue_alive() if args.physical_gpu == 0 else False
        idle_checks = idle_checks + 1 if not apps and not other_queue else 0
        update(state="WAITING_FOR_GPU", stage="resource_wait",
               physical_gpu=args.physical_gpu, gpu_app_pids=apps,
               other_gpu_queue_alive=other_queue,
               consecutive_idle_checks=idle_checks)
        if idle_checks < 3:
            time.sleep(30)
    watcher = tracker = None
    try:
        watcher_log = (RUN / "support_monitor_resume.log").open("a")
        tracker_log = (RUN / "progress_monitor_resume.log").open("a")
        watcher = subprocess.Popen(
            [PF_PY, SCRIPTS / "watch_dior_supports.py", "--shot", "5"],
            cwd=ROOT, stdout=watcher_log, stderr=subprocess.STDOUT)
        tracker = subprocess.Popen(
            [PF_PY, SCRIPTS / "track_table1_progress.py"],
            cwd=ROOT, stdout=tracker_log, stderr=subprocess.STDOUT)
        run(["bash", SCRIPTS / "run_dior_table1_after_grid.sh"], "DIOR_D_5")
        if watcher.poll() is None:
            watcher.wait(timeout=60)
        watcher_log.close()
        run([PF_PY, SCRIPTS / "run_remaining_dior_D.py"], "DIOR_D_10_1")
        run([PF_PY, SCRIPTS / "run_dota_D_after_dior.py"], "DOTA_D")
        update(state="COMPLETE", stage="D_protocol_main_comparison")
    finally:
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
            watcher.wait()
        if tracker is not None and tracker.poll() is None:
            tracker.terminate()
            tracker.wait()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        update(state="FAILED", error=repr(exc))
        print(f"SCQE continuation failed: {exc!r}", file=sys.stderr, flush=True)
        raise
