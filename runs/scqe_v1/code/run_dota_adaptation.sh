#!/usr/bin/env bash
# Adapt CD-ViTO to one frozen DOTA-HBB support episode on SCQE_PHYSICAL_GPU.
# Validation parents/chips/GT are never registered or supplied to this command.
set -euo pipefail
SEED="${1:?seed required}"
SHOT="${2:?shot required}"
case "$SEED" in 2023|42|1234|7|99) ;; *) echo "invalid seed" >&2; exit 2;; esac
case "$SHOT" in 1|5|10) ;; *) echo "invalid shot" >&2; exit 2;; esac
ROOT=/home/root-user/projects/PF_RPN_SR
REPO=$ROOT/third_party/CDFSOD-benchmark
RUN=$ROOT/runs/scqe_v1/backend_dota/seed${SEED}_shot${SHOT}
CONDA=/home/root-user/env/miniconda3/bin/conda
eval "$($CONDA shell.bash hook)"
conda activate cdvito
export DETECTRON2_DATASETS=$ROOT/c0_datasets
export CUDA_HOME=/home/root-user/.conda/envs/cdvito
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SCQE_PHYSICAL_GPU:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
mkdir -p "$RUN"
PROTO=$RUN/DOTA_HBB_${SHOT}shot_seed${SEED}.vitl14.bbox.p${SHOT}.sk.pkl
if [[ ! -f "$PROTO" ]]; then
  python tools/extract_instance_prototypes.py \
    --dataset DOTA_HBB_${SHOT}shot_seed${SEED} --out_dir "$RUN" \
    --model vitl14 --epochs 1 --use_bbox yes --without_mask True
  python tools/run_sinkhorn_cluster.py \
    --inp "$RUN/DOTA_HBB_${SHOT}shot_seed${SEED}.vitl14.bbox.pkl" \
    --epochs 30 --momentum 0.002 --num_prototypes "$SHOT"
fi
if [[ ! -f "$RUN/adapt/model_final.pth" ]]; then
  python tools/train_net.py --num-gpus 1 \
    --config-file configs/dota_hbb/vitl_shot${SHOT}_dota_hbb_finetune.yaml \
    DATASETS.TRAIN "('DOTA_HBB_${SHOT}shot_seed${SEED}',)" \
    DATASETS.TEST "()" TEST.EVAL_PERIOD 0 \
    DE.CLASS_PROTOTYPES "$PROTO" \
    MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth \
    DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    OUTPUT_DIR "$RUN/adapt/"
fi
sha256sum "$PROTO" "$RUN/adapt/model_final.pth" > "$RUN/adaptation_artifacts.sha256"
echo "DOTA-HBB adaptation complete: seed $SEED, shot $SHOT, physical GPU ${SCQE_PHYSICAL_GPU:-0}"
