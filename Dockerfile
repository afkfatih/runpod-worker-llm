# RunPod Serverless Worker for GPT-OSS-20B
# Using official vLLM GPT-OSS Docker image
# OpenAI Compatible API

FROM vllm/vllm-openai:gptoss

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV VLLM_USE_FLASHINFER_SAMPLER=0
# Use V0 engine to avoid FlashAttention 3 requirement
ENV VLLM_USE_V1=0
# Use Triton attention backend for non-Hopper GPUs (4090, A100, L40, etc.)
ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# Set CUDA arch for Ampere/Ada (A100=8.0, 4090=8.9, L40=8.9)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install additional dependencies for RunPod
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    openai-harmony

# Create working directory
WORKDIR /app

# Copy application files
COPY handler.py .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Environment variables for model configuration
# Optimized for RTX A4500 (20GB) + 62GB RAM - FULL 131K CONTEXT
ENV MODEL_NAME="openai/gpt-oss-20b"
ENV MAX_MODEL_LEN=131072
ENV GPU_MEMORY_UTILIZATION=0.98
ENV MAX_NUM_SEQS=16
ENV TENSOR_PARALLEL_SIZE=1
ENV DTYPE="auto"
ENV TRUST_REMOTE_CODE=true
ENV ENABLE_CHUNKED_PREFILL=true
ENV MAX_CONCURRENCY=50
ENV DISABLE_LOG_STATS=false
# CPU offload - use 32GB system RAM for KV cache overflow
ENV CPU_OFFLOAD_GB=32
# Swap space for additional memory (GB)
ENV SWAP_SPACE=8

# Expose port for local testing
EXPOSE 8000

# Override ENTRYPOINT to run our handler instead of vLLM server
ENTRYPOINT []
CMD ["/usr/bin/python3", "-u", "handler.py"]
