#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
PY=/home/root-user/.conda/envs/sqr/bin/python
REPLAY=$ROOT/runs/scqe_v1/code/scripts/replay_b0_backend.py
SOURCE=$RUN/runs/dior_D_5_B0_seed2023
AUDIT=$RUN/audit_attested/data_audit.json

# The first PF-RPN producer occupies GPU0. Wait for its immutable 4,900
# proposal jobs before loading a backend model on the same physical device.
while [ ! -f "$SOURCE/producer_done.json" ]; do
  sleep 5
done

# Check that replaying the same seed on physical GPU0 reproduces the GPU1
# development predictions to numerical precision.
"$PY" "$REPLAY" \
  --source "$RUN/calibration/B0_seed2023_retry1" \
  --seed 2023 --shot 5 --device cuda:0 --audit "$AUDIT" \
  --out "$RUN/calibration/B0_seed2023_gpu0_replay"

"$PY" "$ROOT/runs/scqe_v1/code/scripts/compare_b0_replay.py" \
  --reference "$RUN/calibration/B0_seed2023_retry1/predictions_B0.json" \
  --replay "$RUN/calibration/B0_seed2023_gpu0_replay/predictions_B0.json" \
  --out "$RUN/calibration/B0_replay_device_check.json"

for seed in 42 1234 7; do
  "$PY" "$REPLAY" --source "$SOURCE" --seed "$seed" --shot 5 \
    --device cuda:0 --audit "$AUDIT" \
    --out "$RUN/runs/dior_D_5_B0_seed$seed"
done
