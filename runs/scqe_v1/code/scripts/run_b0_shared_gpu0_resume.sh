#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
PY=/home/root-user/.conda/envs/sqr/bin/python

# Preserve the failed exact-equality diagnostic. The measured GPU-1/GPU-0
# variation must be no larger than two repeat runs on GPU 0 on the frozen
# development 40, using the same proposals, support, and checkpoint hashes.
"$PY" - "$RUN/calibration/B0_replay_variability.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if not d["cross_within_repeat_envelope"]:
    raise SystemExit("cross-device variation exceeds same-device repeat envelope")
if d["cross_AP_difference_pp"] > 0.05:
    raise SystemExit("cross-device development AP drift exceeds 0.05 pp")
PY

for seed in 42 1234 7; do
  "$PY" "$ROOT/runs/scqe_v1/code/scripts/replay_b0_backend.py" \
    --source "$RUN/runs/dior_D_5_B0_seed2023" \
    --seed "$seed" --shot 5 --device cuda:0 \
    --audit "$RUN/audit_attested/data_audit.json" \
    --out "$RUN/runs/dior_D_5_B0_seed$seed"
done
