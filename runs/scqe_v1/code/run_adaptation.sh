#!/usr/bin/env bash
# SCQE backend adaptation: per support seed -> CD-ViTO prototypes -> shared adaptation.
# Usage: bash run_adaptation.sh SEED [SHOT]
# Outputs under runs/scqe_v1/backend/seed{SEED}_shot{SHOT}/ ; adaptation state saved with sha256.
set -e
SEED="${1:?seed required}"
SHOT="${2:-5}"
ROOT=/home/root-user/projects/PF_RPN_SR
REPO=$ROOT/third_party/CDFSOD-benchmark
RUN=$ROOT/runs/scqe_v1/backend/seed${SEED}_shot${SHOT}
CONDA=/home/root-user/env/miniconda3/bin/conda
eval "$($CONDA shell.bash hook)"
conda activate cdvito
export DETECTRON2_DATASETS=$ROOT/c0_datasets
export CUDA_HOME=/home/root-user/.conda/envs/cdvito
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$REPO"
mkdir -p "$RUN"

PROTO=$RUN/DIOR_${SHOT}shot_seed${SEED}.vitl14.bbox.p${SHOT}.sk.pkl
if [ ! -f "$PROTO" ]; then
    echo "--- extract prototypes seed $SEED shot $SHOT"
    python tools/extract_instance_prototypes.py --dataset DIOR_${SHOT}shot_seed${SEED} \
        --out_dir "$RUN" --model vitl14 --epochs 1 \
        --use_bbox yes --without_mask True
    python tools/run_sinkhorn_cluster.py \
        --inp "$RUN/DIOR_${SHOT}shot_seed${SEED}.vitl14.bbox.pkl" \
        --epochs 30 --momentum 0.002 --num_prototypes "$SHOT"
fi

if [ ! -f "$RUN/adapt/model_final.pth" ]; then
    echo "--- adaptation seed $SEED shot $SHOT"
    python tools/train_net.py --num-gpus 1 \
        --config-file configs/dior/vitl_shot${SHOT}_dior_finetune.yaml \
        DATASETS.TRAIN "('DIOR_${SHOT}shot_seed${SEED}',)" \
        DATASETS.TEST "('DIOR_dev100',)" \
        DE.CLASS_PROTOTYPES "$PROTO" \
        MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth \
        DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR "$RUN/adapt/"
fi
sha256sum "$RUN/adapt/model_final.pth" | tee "$RUN/adapt/model_final.sha256"
echo "--- done seed $SEED shot $SHOT"
