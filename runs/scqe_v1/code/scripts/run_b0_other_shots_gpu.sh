#!/usr/bin/env bash
# Train support-specific backends, then evaluate formal B0 on one physical GPU.
set -euo pipefail
device=${1:?physical GPU index 0 or 1 required}
shift
if [ "$device" != 0 ] && [ "$device" != 1 ]; then
  echo "device must be 0 or 1" >&2
  exit 2
fi
ROOT=/home/root-user/projects/PF_RPN_SR
RUN=$ROOT/runs/scqe_server_20261014
SOURCE=$RUN/runs/dior_D_5_B0_seed2023
for spec in "$@"; do
  shot=${spec%%:*}
  seed=${spec#*:}
  case "$shot" in 1|10) ;; *) echo "invalid shot: $shot" >&2; exit 2;; esac
  case "$seed" in 2023|42|1234|7|99) ;; *) echo "invalid seed: $seed" >&2; exit 2;; esac
  echo "starting adaptation: shot=$shot seed=$seed device=$device"
  CUDA_VISIBLE_DEVICES=$device bash "$ROOT/runs/scqe_v1/code/run_adaptation.sh" \
    "$seed" "$shot" > "$RUN/adaptation_logs/seed${seed}_shot${shot}.log" 2>&1
  echo "starting formal B0 replay: shot=$shot seed=$seed device=$device"
  episode=$RUN/runs/dior_D_${shot}_B0_seed$seed
  /home/root-user/.conda/envs/sqr/bin/python \
    "$ROOT/runs/scqe_v1/code/scripts/replay_b0_backend.py" \
    --source "$SOURCE" --seed "$seed" --shot "$shot" \
    --device "cuda:$device" \
    --audit "$RUN/audit_attested/data_audit.json" --out "$episode"
  /home/root-user/.conda/envs/cdvito/bin/python \
    "$ROOT/runs/scqe_v1/code/scripts/evaluate_b0_single.py" \
    --run "$episode" --gt "$RUN/audit_attested/dior_final_gt.json" \
    --audit "$RUN/audit_attested/data_audit.json" \
    --out "$RUN/metrics/per_support/B0_shot${shot}_seed${seed}.json"
done
