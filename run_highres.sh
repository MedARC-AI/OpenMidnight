#!/usr/bin/env bash
# High-resolution fine-tuning stage (Midnight paper, Section 2).
#
# Run this AFTER standard-resolution training (run_1node.sh) has produced a
# teacher checkpoint.  Typical path:
#   output_vitg14/eval/training_<iter>/teacher_checkpoint.pth
#
# Usage:
#   1. Set STAGE1_CHECKPOINT below to the path of your stage-1 teacher checkpoint.
#   2. bash run_highres.sh
#
# To resume an interrupted high-res run, set RESUME="True".

set -euo pipefail

# ── Distributed setup ────────────────────────────────────────────────────────
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29501          # different from stage-1 default (29500)

export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NODE_RANK=0

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4_highres.yaml"
OUTPUT_DIR="./output_vitg14_highres"

# Path to the teacher_checkpoint.pth saved at the end of stage-1 training.
# Example: ./output_vitg14/eval/training_149999/teacher_checkpoint.pth
STAGE1_CHECKPOINT="./output_vitg14/eval/training_149999/teacher_checkpoint.pth"

# Set to "True" to resume a previously started high-res run from its last
# FSDP checkpoint; set to "False" to start fresh from STAGE1_CHECKPOINT.
RESUME="False"

# ── Setup ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RESUME}" == "True" ]]; then
    echo "Resume mode: continuing from last FSDP checkpoint in ${OUTPUT_DIR}"
    RESUME_FLAG=""
    # When resuming we do NOT pass pretrained_weights; the FSDP checkpoint
    # already contains the full model state.
    PRETRAINED_OVERRIDE=""
else
    echo "Fresh start: loading backbone from ${STAGE1_CHECKPOINT}"
    if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
        echo "ERROR: STAGE1_CHECKPOINT not found: ${STAGE1_CHECKPOINT}" >&2
        exit 1
    fi
    rm -rf "${OUTPUT_DIR}"
    RESUME_FLAG="--no-resume"
    PRETRAINED_OVERRIDE="train.pretrained_weights=${STAGE1_CHECKPOINT}"
fi
mkdir -p "${OUTPUT_DIR}"

# ── Launch ───────────────────────────────────────────────────────────────────
uv run torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    dinov2/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${PRETRAINED_OVERRIDE} \
    ${RESUME_FLAG}
