#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
PY=/home/root-user/.conda/envs/cdvito/bin/python
mkdir -p "$RUN/metrics/per_support"

for seed in 2023 42 1234 7 99; do
  episode=$RUN/runs/dior_D_5_B0_seed$seed
  while true; do
    if [ -f "$episode/launch.json" ]; then
      status=$(/home/root-user/.conda/envs/sqr/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$episode/launch.json")
      if [ "$status" = COMPLETE ]; then
        break
      fi
      if [ "$status" = FAILED ]; then
        echo "seed $seed failed; no formal AP will be recorded" >&2
        exit 1
      fi
    fi
    sleep 5
  done
  "$PY" "$ROOT/runs/scqe_v1/code/scripts/evaluate_b0_single.py" \
    --run "$episode" \
    --gt "$RUN/audit_attested/dior_final_gt.json" \
    --audit "$RUN/audit_attested/data_audit.json" \
    --out "$RUN/metrics/per_support/B0_seed$seed.json"
  /home/root-user/.conda/envs/sqr/bin/python \
    "$ROOT/runs/scqe_v1/code/scripts/record_table1_status.py"
done
