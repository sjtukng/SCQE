#!/usr/bin/env bash
# Complete D calibration and the 5-shot B1/B2/B3 comparison without VJP.
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
SCRIPTS=$ROOT/runs/scqe_v1/code/scripts
SPLITS=$ROOT/runs/scqe_v1/splits
PF_PY=/home/root-user/.conda/envs/sqr/bin/python
BE_PY=/home/root-user/.conda/envs/cdvito/bin/python
ACCEPT=$RUN/acceptance_ref64/implementation_acceptance.json
AUDIT=$RUN/audit_attested/data_audit.json
deadline=$(( $(date +%s) + 36000 ))

# The two physical GPUs are occupied by B0 1/10-shot work until both
# aggregates are written. Never overlap another PF/backend model with it.
while [ ! -f "$RUN/metrics/dior_D_1_B0/metrics.json" ] || \
      [ ! -f "$RUN/metrics/dior_D_10_B0/metrics.json" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "B0 1/10-shot work did not finish within ten hours" >&2
    exit 1
  fi
  sleep 30
done

"$PF_PY" "$SCRIPTS/calibration_nograd_scales.py" \
  --split "$SPLITS/dior_dcal40.json" --acceptance "$ACCEPT" \
  --out "$RUN/calibration/nograd_scales"
"$PF_PY" "$SCRIPTS/calibration_nograd_grid.py" \
  --scales "$RUN/calibration/nograd_scales/scales.json" \
  --split "$SPLITS/dior_dcal40.json" --acceptance "$ACCEPT" \
  --out "$RUN/calibration/nograd_grid"
bash "$RUN/calibration/nograd_grid/commands.sh"
"$BE_PY" "$SCRIPTS/calibration_nograd_select.py" \
  --trials "$RUN/calibration/nograd_grid/trial_manifest.json" \
  --scales "$RUN/calibration/nograd_scales/scales.json" \
  --gt "$SPLITS/dcal40_gt.json" --out "$RUN/config_lock_D_nograd.json"

for seed in 2023 42 1234; do
  "$PF_PY" "$SCRIPTS/run_dior_pair.py" \
    --split "$SPLITS/dior_dcheck60.json" --seed "$seed" --shot 5 \
    --methods B0,B1,B2,B3 --lock "$RUN/config_lock_D_nograd.json" \
    --acceptance "$ACCEPT" --out "$RUN/confirmation/nograd_seed$seed" \
    --record-state
done
"$BE_PY" "$SCRIPTS/confirm_nograd_calibration.py" \
  --run "2023:$RUN/confirmation/nograd_seed2023" \
  --run "42:$RUN/confirmation/nograd_seed42" \
  --run "1234:$RUN/confirmation/nograd_seed1234" \
  --gt "$SPLITS/dcheck60_gt.json" \
  --out "$RUN/confirmation/nograd_metrics.json"

for seed in 2023 42 1234 7 99; do
  "$PF_PY" "$SCRIPTS/run_dior_pair.py" \
    --split "$SPLITS/dior_final_queries.json" \
    --seed "$seed" --shot 5 --methods B1,B2,B3 \
    --lock "$RUN/config_lock_D_nograd.json" \
    --acceptance "$ACCEPT" --audit "$AUDIT" \
    --out "$RUN/runs/dior_D_5_nograd_seed$seed" --record-state
done
"$BE_PY" "$SCRIPTS/evaluate_runs.py" \
  --protocol D --dataset DIOR --shot 5 \
  --gt "$RUN/audit_attested/dior_final_gt.json" \
  --run "2023:$RUN/runs/dior_D_5_nograd_seed2023" \
  --run "42:$RUN/runs/dior_D_5_nograd_seed42" \
  --run "1234:$RUN/runs/dior_D_5_nograd_seed1234" \
  --run "7:$RUN/runs/dior_D_5_nograd_seed7" \
  --run "99:$RUN/runs/dior_D_5_nograd_seed99" \
  --out "$RUN/metrics/dior_D_5_nograd"
"$PF_PY" "$SCRIPTS/record_table1_status.py"
"$PF_PY" "$SCRIPTS/build_report.py" --run-root "$RUN"
echo "B1/B2/B3 D DIOR 5-shot formal pipeline complete"
