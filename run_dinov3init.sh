#!/usr/bin/env bash
# run_dinov3init.sh — fine-tune from an external / DINOv3-format checkpoint.
#
# Use this when your starting weights come from:
#   • A prior OpenMidnight teacher_checkpoint.pth
#   • A path-fm-dinov3 (MedARC-AI) backbone checkpoint
#   • Any flat backbone .pth or {'model': ...} format file
#
# The loader auto-detects the checkpoint format and loads backbone weights with
# strict=False, so architecture-specific keys (e.g. DINOv3 RoPE tensors) that
# are absent from this model are safely skipped.
#
# Usage:
#   STAGE1_CHECKPOINT=/path/to/teacher_checkpoint.pth bash run_dinov3init.sh
#
# You MUST set STAGE1_CHECKPOINT.

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12357}

: "${STAGE1_CHECKPOINT:?Please set STAGE1_CHECKPOINT to the path of your pretrained checkpoint.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/dinov2/configs/train/vitg14_reg4_dinov3init.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-/block/openmidnight-dinov3init-$(date +%Y%m%d_%H%M%S)}"

echo "=== OpenMidnight: external-checkpoint fine-tuning ==="
echo "  GPUs       : ${NUM_GPUS}"
echo "  Config     : ${CONFIG_FILE}"
echo "  Checkpoint : ${STAGE1_CHECKPOINT}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/dinov2/train/train.py" \
    --config-file "${CONFIG_FILE}" \
    --no-resume \
    train.output_dir="${OUTPUT_DIR}" \
    train.pretrained_weights="${STAGE1_CHECKPOINT}"
