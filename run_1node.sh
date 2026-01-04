#!/usr/bin/env bash
set -euo pipefail

# Master node configuration 
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

export NNODES=1 # number of nodes you are using
export NPROC_PER_NODE=8 # number of GPUs per node
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # specific devices to use on this node
export NODE_RANK=0 # the node running this script will be master node (rank 0)

# Training config
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_BRCA"
RESUME="False" # set string to "True" to resume from last checkpoint in OUTPUT_DIR if it exists

# Set Python path for imports
# Provide script path so train.py can attach the right launcher script to WandB.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Clean output directory only when not resuming
if [[ "${RESUME}" == "True" ]]; then
  echo "Resume enabled; preserving ${OUTPUT_DIR}"
  RESUME_FLAG=""
else
  echo "Resume disabled; cleaning ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
  RESUME_FLAG="--no-resume"
fi
mkdir -p "${OUTPUT_DIR}"

echo "[Master Node] Starting training..."
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
