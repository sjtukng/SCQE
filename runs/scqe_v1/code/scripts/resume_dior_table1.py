"""Resume immutable Dcal trials, record AP after each, then launch formal stages."""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
RUN = ROOT / "runs/scqe_server_20261014"
GRID = RUN / "calibration/D_fp64_v4"
SCRIPTS = ROOT / "runs/scqe_v1/code/scripts"
BE_PY = "/home/root-user/.conda/envs/cdvito/bin/python"
STATE = RUN / "table1_pipeline_state.json"


def update(**fields):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(timestamp=time.time(), **fields)
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE)


def main():
    rows = json.loads((GRID / "trial_manifest.json").read_text())
    commands = (GRID / "commands.sh").read_text().splitlines()[2:]
    if len(rows) != 54 or len(commands) != 54:
        raise RuntimeError("expected complete frozen 27x2 grid manifest")
    update(status="RUNNING", stage="E3_grid", completed=0, total=len(rows))
    for index, (row, command) in enumerate(zip(rows, commands), 1):
        out = ROOT / row["run_dir"]
        if out.exists():
            launch = json.loads((out / "launch.json").read_text())
            if launch.get("status") != "COMPLETE":
                raise RuntimeError(f"immutable trial requires investigation: {out}")
            print(f"[{index}/54] already complete: {out.name}", flush=True)
        else:
            print(f"[{index}/54] running: {out.name}", flush=True)
            subprocess.run(shlex.split(command), cwd=ROOT, check=True)
        subprocess.run([
            BE_PY, str(SCRIPTS / "calibration_progress.py"),
            "--trials", str(GRID / "trial_manifest.json"),
            "--gt", str(ROOT / "runs/scqe_v1/splits/dcal40_gt.json"),
            "--out", str(GRID / "progress.json"),
        ], cwd=ROOT, check=True)
        update(status="RUNNING", stage="E3_grid", completed=index, total=len(rows))
    update(status="RUNNING", stage="E3_lock_confirmation_formal")
    subprocess.run(["bash", str(SCRIPTS / "run_dior_table1_after_grid.sh")],
                   cwd=ROOT, check=True)
    update(status="COMPLETE", stage="DIOR_D_5_table1")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        update(status="FAILED", error=repr(exc))
        raise
