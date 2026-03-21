#!/usr/bin/env bash
# run_scratch.sh — launch from-scratch ViT-G/14 training (no pretrained init).
#
# This script trains the model from random initialisation using the tuned
# hyperparameters in vitg14_reg4_scratch.yaml (higher LR, longer warmup,
# layerwise decay 0.9).
#
# Usage:
#   bash run_scratch.sh
#
# Adjust NUM_GPUS, OUTPUT_DIR, and data paths as needed.

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12355}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/dinov2/configs/train/vitg14_reg4_scratch.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-/block/openmidnight-scratch-$(date +%Y%m%d_%H%M%S)}"

echo "=== OpenMidnight: from-scratch training ==="
echo "  GPUs        : ${NUM_GPUS}"
echo "  Config      : ${CONFIG_FILE}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/dinov2/train/train.py" \
    --config-file "${CONFIG_FILE}" \
    --no-resume \
    train.output_dir="${OUTPUT_DIR}"
