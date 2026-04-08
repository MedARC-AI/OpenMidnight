#!/usr/bin/env bash
# run_gram.sh — launch ViT-G/14 fine-tuning WITH the DINOv3-style Gram loss.
#
# The Gram loss adds a patch-similarity MSE term (dinov2/loss/gram_loss.py) on
# top of the standard DINO + iBOT objective.  The EMA teacher's patch tokens
# are used as reference (no separate frozen model needed).
#
# Compare results against a baseline run of run_train.sh (vitg14_reg4.yaml) to
# ablate the effect of the Gram objective.
#
# Usage:
#   bash run_gram.sh
#
# Set GRAM_LOSS_WEIGHT to tune the weight of the Gram term (default 0.1).

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12356}
GRAM_LOSS_WEIGHT=${GRAM_LOSS_WEIGHT:-0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/dinov2/configs/train/vitg14_reg4_gram.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-/block/openmidnight-gram-$(date +%Y%m%d_%H%M%S)}"

echo "=== OpenMidnight: DINOv3-style Gram loss training ==="
echo "  GPUs            : ${NUM_GPUS}"
echo "  Config          : ${CONFIG_FILE}"
echo "  Gram loss weight: ${GRAM_LOSS_WEIGHT}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/dinov2/train/train.py" \
    --config-file "${CONFIG_FILE}" \
    --no-resume \
    train.output_dir="${OUTPUT_DIR}" \
    gram.loss_weight="${GRAM_LOSS_WEIGHT}"
