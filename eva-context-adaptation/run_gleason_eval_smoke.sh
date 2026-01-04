# =============================================================================
# GleasonArvaniti Evaluation Smoke Test - Fair Comparison
# =============================================================================
#
# This script runs a small-scale test to verify the evaluation pipeline works
# correctly before running on the full dataset.
#
# It compares:
#   1. BASELINE: OpenMidnight CLS token only (no context)
#   2. CONTEXT: Context-adapted embedding from your trained adapter
#
# Both use the SAME patches from the same positions, ensuring fair comparison.
#
# Usage:
#   ./run_gleason_eval_smoke.sh
#
# Optional environment variables:
#   N_SAMPLES=100          # samples per split (default: 100)
#   CONTEXT_CKPT=...       # path to context adapter checkpoint
#   CONTEXT_CFG=...        # path to context adapter config
#   BASELINE_CKPT=...      # path to OpenMidnight checkpoint
#   SKIP_BASELINE=true     # skip baseline generation
#   SKIP_CONTEXT=true      # skip context generation
# =============================================================================

set -euo pipefail

# Configuration
N_SAMPLES=${N_SAMPLES:-100}

# Data backend
DATA_BACKEND=${DATA_BACKEND:-s3}    # <-- Data source: s3 | local
GLEASON_ROOT=${GLEASON_ROOT:-""}    # <-- Required if DATA_BACKEND=local (download data from S3 bucket)

# AWS S3 settings for GleasonArvaniti dataset
AWS_ENDPOINT=${AWS_ENDPOINT} # <-- Set to your AWS S3 endpoint
AWS_PROFILE=${AWS_PROFILE:-default} # <-- Set to your AWS profile name
AWS_BUCKET=${AWS_BUCKET:-path-datasets}
AWS_GLEASON_ROOT=${AWS_GLEASON_ROOT:-arvaniti_gleason_patches}

# Model checkpoints
BASELINE_CKPT=${BASELINE_CKPT:-/home/paul/OpenMidnight/checkpoints/teacher_epoch250000.pth}
CONTEXT_CKPT=${CONTEXT_CKPT:-/home/paul/OpenMidnight/outputs/context_adaptation/checkpoint_final.pth}
CONTEXT_CFG=${CONTEXT_CFG:-/home/paul/OpenMidnight/dinov2/configs/train/context_adaptation.yaml}

# Output directories
OUTPUT_BASE=${OUTPUT_BASE:-/home/paul/OpenMidnight/outputs/gleason_eval_smoke}
OUTPUT_BASELINE=${OUTPUT_BASELINE:-${OUTPUT_BASE}/baseline}
OUTPUT_CONTEXT=${OUTPUT_CONTEXT:-${OUTPUT_BASE}/context}

# Eval config (use embeddings-only config for pre-computed embeddings)
EVAL_CONFIG=${EVAL_CONFIG:-/home/paul/OpenMidnight/eval_configs/gleason_embeddings_only.yaml}

# Skip flags
SKIP_BASELINE=${SKIP_BASELINE:-true}
SKIP_CONTEXT=${SKIP_CONTEXT:-false}

# Processing options
BATCH_SIZE=${BATCH_SIZE:-16}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-64}
DEVICE=${DEVICE:-cuda}
N_RUNS=${N_RUNS:-1}

# Export for subprocesses
export PYTHONPATH="/home/paul/OpenMidnight:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/gleason_context_eval.py"

echo "=============================================="
echo "GleasonArvaniti Evaluation Smoke Test"
echo "=============================================="
echo "Samples per split: ${N_SAMPLES}"
echo "Data backend:      ${DATA_BACKEND}"
echo "Gleason root:      ${GLEASON_ROOT}"
echo "AWS Bucket:        ${AWS_BUCKET}"
echo "AWS Gleason Root:  ${AWS_GLEASON_ROOT}"
echo "Baseline ckpt:     ${BASELINE_CKPT}"
echo "Context ckpt:      ${CONTEXT_CKPT}"
echo "Context config:    ${CONTEXT_CFG}"
echo "Output baseline:   ${OUTPUT_BASELINE}"
echo "Output context:    ${OUTPUT_CONTEXT}"
echo "Eval config:       ${EVAL_CONFIG}"
echo "Device:            ${DEVICE}"
echo "=============================================="

# Create output directories
mkdir -p "${OUTPUT_BASELINE}"
mkdir -p "${OUTPUT_CONTEXT}"

# -----------------------------------------------------------------------------
# Step 1: Generate baseline embeddings
# -----------------------------------------------------------------------------
if [ "${SKIP_BASELINE}" = "true" ]; then
    echo ""
    echo ">>> SKIPPING baseline embedding generation"
else
    echo ""
    echo ">>> Step 1: Generating BASELINE embeddings (OpenMidnight CLS only)"
    echo "    This uses the exact patches from GleasonArvaniti."
    echo ""

    python "${EVAL_SCRIPT}" \
        --mode baseline \
        --data-backend "${DATA_BACKEND}" \
        --gleason-root "${GLEASON_ROOT}" \
        --baseline-checkpoint "${BASELINE_CKPT}" \
        --aws-endpoint "${AWS_ENDPOINT}" \
        --aws-profile "${AWS_PROFILE}" \
        --aws-bucket "${AWS_BUCKET}" \
        --aws-gleason-root "${AWS_GLEASON_ROOT}" \
        --output-root "${OUTPUT_BASELINE}" \
        --max-samples "${N_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --model-name "baseline_smoke" \
        --n-runs "${N_RUNS}" \
        --eval-config "${EVAL_CONFIG}" \
        2>&1 | tee "${OUTPUT_BASELINE}/log.txt"
fi

# -----------------------------------------------------------------------------
# Step 2: Generate context embeddings
# -----------------------------------------------------------------------------
if [ "${SKIP_CONTEXT}" = "true" ]; then
    echo ""
    echo ">>> SKIPPING context embedding generation"
else
    echo ""
    echo ">>> Step 2: Generating CONTEXT embeddings (context-adapted)"
    echo "    This extracts 16x16 regions from WSIs and uses the context adapter."
    echo ""

    python "${EVAL_SCRIPT}" \
        --mode context \
        --data-backend "${DATA_BACKEND}" \
        --gleason-root "${GLEASON_ROOT}" \
        --checkpoint "${CONTEXT_CKPT}" \
        --config "${CONTEXT_CFG}" \
        --aws-endpoint "${AWS_ENDPOINT}" \
        --aws-profile "${AWS_PROFILE}" \
        --aws-bucket "${AWS_BUCKET}" \
        --aws-gleason-root "${AWS_GLEASON_ROOT}" \
        --output-root "${OUTPUT_CONTEXT}" \
        --max-samples "${N_SAMPLES}" \
        --batch-size 1 \
        --patch-batch-size "${PATCH_BATCH_SIZE}" \
        --device "${DEVICE}" \
        --model-name "context_smoke" \
        --n-runs "${N_RUNS}" \
        --eval-config "${EVAL_CONFIG}" \
        2>&1 | tee "${OUTPUT_CONTEXT}/log.txt"
fi

echo ""
echo "=============================================="
echo "Smoke Test Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Baseline: ${OUTPUT_BASELINE}"
echo "  Context:  ${OUTPUT_CONTEXT}"
echo ""
echo "To compare results, look at the eva fit output in each log.txt file."
echo "Look for 'test/BinaryBalancedAccuracy' metric."
echo ""
