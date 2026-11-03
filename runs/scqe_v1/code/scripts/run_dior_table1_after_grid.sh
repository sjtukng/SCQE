#!/usr/bin/env bash
# Continue the formal DIOR/D/5-shot pipeline only after all Dcal runs finish.
set -euo pipefail

ROOT=/home/root-user/projects/PF_RPN_SR
RUN="$ROOT/runs/scqe_server_20261014"
SCRIPTS="$ROOT/runs/scqe_v1/code/scripts"
SPLITS="$ROOT/runs/scqe_v1/splits"
PF_PY=/home/root-user/.conda/envs/sqr/bin/python
BE_PY=/home/root-user/.conda/envs/cdvito/bin/python
GRID="$RUN/calibration/D_fp64_v4"
SCALES="$RUN/calibration/scales_fp64_v1/scales.json"
ACCEPT="$RUN/acceptance_fp64_vjp_complete/implementation_acceptance.json"
AUDIT="$RUN/audit_attested/data_audit.json"
GT="$RUN/audit_attested/dior_final_gt.json"
cd "$ROOT"

if [[ ! -e "$RUN/config_lock_D.json" ]]; then
  "$BE_PY" "$SCRIPTS/calibration_select.py" \
    --trials "$GRID/trial_manifest.json" --scales "$SCALES" \
    --gt "$SPLITS/dcal40_gt.json" --out "$RUN/config_lock_D.json"
fi

for seed in 2023 42 1234; do
  out="$RUN/confirmation/seed$seed"
  if [[ ! -e "$out" ]]; then
    "$PF_PY" "$SCRIPTS/run_dior_pair.py" \
      --split "$SPLITS/dior_dcheck60.json" --seed "$seed" --shot 5 \
      --methods B0,B1,B2,B3,SCQE --lock "$RUN/config_lock_D.json" \
      --acceptance "$ACCEPT" --out "$out" --record-state --same-job-cache \
      --single-gpu
  fi
  "$PF_PY" - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if json.loads((p / 'launch.json').read_text())['status'] != 'COMPLETE':
    raise RuntimeError(f'incomplete immutable confirmation run: {p}')
PY
done

if [[ ! -e "$RUN/confirmation/metrics.json" ]]; then
  "$BE_PY" "$SCRIPTS/confirm_calibration.py" \
    --run "2023:$RUN/confirmation/seed2023" \
    --run "42:$RUN/confirmation/seed42" \
    --run "1234:$RUN/confirmation/seed1234" \
    --gt "$SPLITS/dcheck60_gt.json" \
    --out "$RUN/confirmation/metrics.json"
fi

for seed in 2023 42 1234 7 99; do
  out="$RUN/runs/dior_D_5_seed$seed"
  resume_args=()
  if [[ "$seed" == 1234 && -e "$out" ]]; then
    status=$("$PF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/launch.json")
    if [[ "$status" != COMPLETE ]]; then
      resume_args=(--resume-from "$out")
      out="$RUN/runs/dior_D_5_seed1234_resume1"
    fi
    if [[ -e "$out" ]]; then
      status=$("$PF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/launch.json")
      if [[ "$status" != COMPLETE && "$out" == *resume1 ]]; then
        resume_args=(--resume-from "$out")
        out="$RUN/runs/dior_D_5_seed1234_resume2"
      fi
    fi
  fi
  # Any formal single-GPU episode may be interrupted by a requested pause.
  # Preserve the immutable source and stage its complete prefix into a fresh
  # resume directory on the next invocation.
  if [[ "$seed" != 1234 && -e "$out" ]]; then
    status=$("$PF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/launch.json")
    if [[ "$status" != COMPLETE ]]; then
      resume_args=(--resume-from "$out")
      out="$RUN/runs/dior_D_5_seed${seed}_resume1"
    fi
    if [[ -e "$out" ]]; then
      status=$("$PF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/launch.json")
      if [[ "$status" != COMPLETE && "$out" == *resume1 ]]; then
        resume_args=(--resume-from "$out")
        out="$RUN/runs/dior_D_5_seed${seed}_resume2"
      fi
    fi
  fi
  if [[ ! -e "$out" ]]; then
    "$PF_PY" "$SCRIPTS/run_dior_pair.py" \
      --split "$SPLITS/dior_final_queries.json" --seed "$seed" --shot 5 \
      --methods B0,B1,B2,B3,SCQE --lock "$RUN/config_lock_D.json" \
      --acceptance "$ACCEPT" --audit "$AUDIT" --out "$out" --record-state \
      --same-job-cache --single-gpu --physical-gpu "${SCQE_PHYSICAL_GPU:-0}" \
      "${resume_args[@]}"
  fi
  "$PF_PY" - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if json.loads((p / 'launch.json').read_text())['status'] != 'COMPLETE':
    raise RuntimeError(f'incomplete immutable formal run: {p}')
PY
done

seed1234_out="$RUN/runs/dior_D_5_seed1234"
if [[ -e "$RUN/runs/dior_D_5_seed1234_resume1" ]]; then
  seed1234_out="$RUN/runs/dior_D_5_seed1234_resume1"
fi
if [[ -e "$RUN/runs/dior_D_5_seed1234_resume2" ]]; then
  seed1234_out="$RUN/runs/dior_D_5_seed1234_resume2"
fi
seed99_out="$RUN/runs/dior_D_5_seed99"
if [[ -e "$RUN/runs/dior_D_5_seed99_resume1" ]]; then
  seed99_out="$RUN/runs/dior_D_5_seed99_resume1"
fi
if [[ -e "$RUN/runs/dior_D_5_seed99_resume2" ]]; then
  seed99_out="$RUN/runs/dior_D_5_seed99_resume2"
fi
common=(
  --protocol D --dataset DIOR --shot 5 --gt "$GT"
  --run "2023:$RUN/runs/dior_D_5_seed2023"
  --run "42:$RUN/runs/dior_D_5_seed42"
  --run "1234:$seed1234_out"
  --run "7:$RUN/runs/dior_D_5_seed7"
  --run "99:$seed99_out"
)
if [[ ! -e "$RUN/metrics/dior_D_5_primary" ]]; then
  "$BE_PY" "$SCRIPTS/evaluate_runs.py" "${common[@]}" \
    --out "$RUN/metrics/dior_D_5_primary"
fi
"$PF_PY" "$SCRIPTS/record_table1_status.py"
if [[ ! -e "$RUN/metrics/dior_D_5" ]]; then
  "$BE_PY" "$SCRIPTS/evaluate_runs.py" "${common[@]}" --bootstrap \
    --out "$RUN/metrics/dior_D_5"
fi
"$PF_PY" "$SCRIPTS/record_table1_status.py"
"$PF_PY" "$SCRIPTS/build_report.py" --run-root "$RUN"
