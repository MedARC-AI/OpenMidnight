#!/usr/bin/env bash
#SBATCH --job-name=dinov2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --time=infinite
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --partition=main
#SBATCH --account=training

set -euo pipefail

# ---- Training config ----
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="/data/rename_this_to_your_output_dir_name"
RESUME="True"   # "True" to keep OUTPUT_DIR and resume if checkpoints exist

# ---- Repo env ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Clean output directory
if [[ "${RESUME}" == "True" ]]; then
  RESUME_FLAG=""
else
  RESUME_FLAG="--no-resume"
  if [[ "${SLURM_PROCID}" == "0" ]]; then
    rm -rf "${OUTPUT_DIR}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG}
