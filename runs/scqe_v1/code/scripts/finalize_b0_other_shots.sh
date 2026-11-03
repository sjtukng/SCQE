#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
BE_PY=/home/root-user/.conda/envs/cdvito/bin/python
PF_PY=/home/root-user/.conda/envs/sqr/bin/python
deadline=$(( $(date +%s) + 36000 ))

for shot in 1 10; do
  for seed in 2023 42 1234 7 99; do
    metric=$RUN/metrics/per_support/B0_shot${shot}_seed${seed}.json
    while [ ! -f "$metric" ]; do
      if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "timed out waiting for $metric" >&2
        exit 1
      fi
      sleep 15
    done
  done
  "$BE_PY" "$ROOT/runs/scqe_v1/code/scripts/evaluate_runs.py" \
    --protocol D --dataset DIOR --shot "$shot" \
    --gt "$RUN/audit_attested/dior_final_gt.json" \
    --run "2023:$RUN/runs/dior_D_${shot}_B0_seed2023" \
    --run "42:$RUN/runs/dior_D_${shot}_B0_seed42" \
    --run "1234:$RUN/runs/dior_D_${shot}_B0_seed1234" \
    --run "7:$RUN/runs/dior_D_${shot}_B0_seed7" \
    --run "99:$RUN/runs/dior_D_${shot}_B0_seed99" \
    --out "$RUN/metrics/dior_D_${shot}_B0"
  "$PF_PY" "$ROOT/runs/scqe_v1/code/scripts/record_table1_status.py"
done

"$PF_PY" "$ROOT/runs/scqe_v1/code/scripts/summarize_adaptation.py"
"$PF_PY" "$ROOT/runs/scqe_v1/code/scripts/build_report.py" --run-root "$RUN"
echo "B0 1/5/10-shot DIOR D finalization complete"
