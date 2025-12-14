#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.
# Incremental training script for DIDA framework
# Usage: bash train_incremental.sh <experiment_name> <gpu_ids>

set -e

EXPERIMENT=$1
GPU_IDS=${2:-"0"}
WORK_DIR="work_dirs/${EXPERIMENT}"

echo "========================================="
echo "DIDA Incremental Training"
echo "Experiment: ${EXPERIMENT}"
echo "GPUs: ${GPU_IDS}"
echo "========================================="

# Stage 0: Source domain
echo ""
echo "Stage 0: Training on source domain..."
python tools/train.py \
    projects/dida/configs/${EXPERIMENT}/stage0_voc.py \
    --work-dir ${WORK_DIR}/stage0 \
    --cfg-options env_cfg.cudnn_benchmark=True \
    CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Stage 1: First target domain
echo ""
echo "Stage 1: Adapting to first target domain..."
python tools/train.py \
    projects/dida/configs/${EXPERIMENT}/stage1_clipart.py \
    --work-dir ${WORK_DIR}/stage1 \
    --cfg-options \
        load_from=${WORK_DIR}/stage0/epoch_12.pth \
        env_cfg.cudnn_benchmark=True \
    CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Stage 2: Second target domain
echo ""
echo "Stage 2: Adapting to second target domain..."
python tools/train.py \
    projects/dida/configs/${EXPERIMENT}/stage2_watercolor.py \
    --work-dir ${WORK_DIR}/stage2 \
    --cfg-options \
        load_from=${WORK_DIR}/stage1/epoch_12.pth \
        env_cfg.cudnn_benchmark=True \
    CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Stage 3: Third target domain
echo ""
echo "Stage 3: Adapting to third target domain..."
python tools/train.py \
    projects/dida/configs/${EXPERIMENT}/stage3_comic.py \
    --work-dir ${WORK_DIR}/stage3 \
    --cfg-options \
        load_from=${WORK_DIR}/stage2/epoch_12.pth \
        env_cfg.cudnn_benchmark=True \
    CUDA_VISIBLE_DEVICES=${GPU_IDS}

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved in: ${WORK_DIR}"
echo "========================================="
