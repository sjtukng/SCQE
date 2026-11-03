#!/usr/bin/env bash
# Continue the audited DIOR D 5-shot B0 baseline after seed 2023 completes.
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
PY=/home/root-user/.conda/envs/sqr/bin/python
FIRST=$RUN/runs/dior_D_5_B0_seed2023/launch.json

while true; do
  status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$FIRST")
  if [ "$status" = COMPLETE ]; then
    break
  fi
  if [ "$status" = FAILED ]; then
    echo "seed 2023 failed; refusing to continue" >&2
    exit 1
  fi
  sleep 5
done

for seed in 42 1234 7 99; do
  "$PY" "$ROOT/runs/scqe_v1/code/scripts/run_dior_pair.py" \
    --split "$ROOT/runs/scqe_v1/splits/dior_final_queries.json" \
    --seed "$seed" --shot 5 --methods B0 \
    --lock "$RUN/config_lock_B0_D.json" \
    --acceptance "$RUN/acceptance_ref64/implementation_acceptance.json" \
    --audit "$RUN/audit_attested/data_audit.json" \
    --out "$RUN/runs/dior_D_5_B0_seed$seed"
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
