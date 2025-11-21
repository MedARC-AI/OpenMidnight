#!/usr/bin/env bash
set -euo pipefail

# Worker node configuration
export MASTER_ADDR="10.128.0.10"  # <-- Replace this with master node IP
export MASTER_PORT=29500

export NNODES=2
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # specific devices to use on this node
export NODE_RANK=$(( NODE_RANK + 1 )) # non-master node rank (1 for first worker, etc.)

# Training config (must match master node! double-check your run.sh script!)
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_vitg14"
RESUME="True" # set string to "True" to resume from last checkpoint in OUTPUT_DIR if it exists

# Set Python path for imports
# Provide script path so train.py can attach the right launcher script to WandB.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Clean output directory only when not resuming
if [[ "${RESUME}" == "True" ]]; then
  RESUME_FLAG=""
else
  RESUME_FLAG="--no-resume"
fi

echo "[Worker Node ${NODE_RANK}] Joining training..."
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG} 
