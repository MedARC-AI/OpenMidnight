#!/bin/bash
# Run Context Adaptation Training for OpenMidnight
# Trains a context-ViT to process 3584x3584 regions using frozen OpenMidnight patch embeddings
# Uses 8 H100 GPUs on a single node

set -e

# Configuration
CONFIG_FILE="/home/paul/OpenMidnight/dinov2/configs/train/context_adaptation.yaml"
OUTPUT_DIR="/home/paul/OpenMidnight/outputs/context_adaptation"
NUM_GPUS=8
RESUME="False" # set to "True" to resume from existing OUTPUT_DIR/wandb_run_id.txt

# Handle resume / fresh start
if [[ "${RESUME}" == "True" ]]; then
    echo "Resume enabled; preserving ${OUTPUT_DIR}"
else
    echo "Resume disabled; cleaning ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/debug_images"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONPATH="/home/paul/OpenMidnight:$PYTHONPATH"

# Enable TF32 for better performance on H100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "=========================================="
echo "Context Adaptation Training for OpenMidnight"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Run distributed training with torchrun
uv run torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    /home/paul/OpenMidnight/dinov2/train/train_context.py \
    "$CONFIG_FILE"

echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Debug images saved to: $OUTPUT_DIR/debug_images"
echo "=========================================="
