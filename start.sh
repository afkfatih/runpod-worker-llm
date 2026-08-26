#!/bin/bash
# Start script for RunPod Serverless Worker
# Optimized for GPT-OSS-20B inference

set -e

echo "=========================================="
echo "RunPod GPT-OSS-20B Worker Starting..."
echo "=========================================="

# Environment info
echo "Model: ${MODEL_NAME:-openai/gpt-oss-20b}"
echo "Max Context Length: ${MAX_MODEL_LEN:-32768}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.95}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE:-1}"

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')" || true

# Check GPU memory
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

echo "=========================================="
echo "Starting vLLM inference engine..."
echo "=========================================="

# huggingface_hub reads HF_TOKEN from the environment on its own; passing it on a
# command line would expose it in the container's process list.
if [ ! -z "${HF_TOKEN}" ]; then
    echo "HuggingFace token detected, will be used for gated models."
fi

# Start the handler
exec python -u handler.py

