#!/usr/bin/env bash
#SBATCH --job-name=dinov2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --time=infinite
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err
#SBATCH --partition=main
#SBATCH --exclusive
#SBATCH --account=training

set -euo pipefail

# ---- Training config ----
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="/data/ratna_retrain" # RSG - changed this to my dir
RESUME="False"   # "True" to keep OUTPUT_DIR and resume if checkpoints exist # RSG - set to False

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

# RSG - avoid issues with the default cache dir being shared across jobs and filling up the disk
export TORCH_HOME="/tmp/${USER}/torch/${SLURM_JOB_ID}"
export TRITON_HOME="/tmp/${USER}/triton/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
mkdir -p "${TORCH_HOME}" "${TRITON_CACHE_DIR}"

echo "Prewarming torch.hub into ${TORCH_HOME}"
uv run python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', trust_repo=True)"


uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG}
