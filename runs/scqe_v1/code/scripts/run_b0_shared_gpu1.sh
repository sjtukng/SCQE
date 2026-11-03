#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
PY=/home/root-user/.conda/envs/sqr/bin/python
REPLAY=$ROOT/runs/scqe_v1/code/scripts/replay_b0_backend.py
SOURCE=$RUN/runs/dior_D_5_B0_seed2023
AUDIT=$RUN/audit_attested/data_audit.json

while [ ! -f "$SOURCE/worker_done.json" ]; do
  status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SOURCE/launch.json")
  if [ "$status" = FAILED ]; then
    echo "seed2023 failed; refusing to continue" >&2
    exit 1
  fi
  sleep 5
done

"$PY" "$REPLAY" --source "$SOURCE" --seed 99 --shot 5 \
  --device cuda:1 --audit "$AUDIT" \
  --out "$RUN/runs/dior_D_5_B0_seed99"

for seed in 42 1234 7; do
  path=$RUN/runs/dior_D_5_B0_seed$seed/launch.json
  while true; do
    if [ -f "$path" ]; then
      status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$path")
      if [ "$status" = COMPLETE ]; then
        break
      fi
      if [ "$status" = FAILED ]; then
        echo "seed$seed replay failed; refusing to evaluate" >&2
        exit 1
      fi
    fi
    sleep 5
  done
done

/home/root-user/.conda/envs/cdvito/bin/python \
  "$ROOT/runs/scqe_v1/code/scripts/evaluate_runs.py" \
  --protocol D --dataset DIOR --shot 5 \
  --gt "$RUN/audit_attested/dior_final_gt.json" \
  --run "2023:$RUN/runs/dior_D_5_B0_seed2023" \
  --run "42:$RUN/runs/dior_D_5_B0_seed42" \
  --run "1234:$RUN/runs/dior_D_5_B0_seed1234" \
  --run "7:$RUN/runs/dior_D_5_B0_seed7" \
  --run "99:$RUN/runs/dior_D_5_B0_seed99" \
  --out "$RUN/metrics/dior_D_5_B0"

"$PY" "$ROOT/runs/scqe_v1/code/scripts/record_table1_status.py"
"$PY" "$ROOT/runs/scqe_v1/code/scripts/build_report.py" --run-root "$RUN"
