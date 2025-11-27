# RunPod Serverless Worker for GPT-OSS-20B
# OpenAI Compatible vLLM Inference Engine
# Optimized for fastest inference with full context

# CUDA 12.8 required for vLLM GPT-OSS wheels
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install uv for faster package management
RUN pip install --upgrade pip setuptools wheel
RUN pip install uv

# Install vLLM with GPT-OSS support (special version for CUDA 12.8)
RUN uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install additional dependencies
RUN uv pip install \
    runpod>=1.7.0 \
    openai-harmony \
    huggingface_hub \
    transformers \
    tokenizers \
    accelerate \
    safetensors \
    aiohttp \
    uvicorn \
    fastapi

# Production stage - use runtime with CUDA 12.8
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# Install runtime dependencies with deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create working directory
WORKDIR /app

# Copy application files
COPY handler.py .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Environment variables for model configuration
ENV MODEL_NAME="openai/gpt-oss-20b"
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.95
ENV MAX_NUM_SEQS=256
ENV TENSOR_PARALLEL_SIZE=1
ENV DTYPE="auto"
ENV TRUST_REMOTE_CODE=true
ENV ENABLE_CHUNKED_PREFILL=true
ENV MAX_CONCURRENCY=300
ENV DISABLE_LOG_STATS=false

# Expose port for local testing
EXPOSE 8000

# Start the worker
CMD ["python", "-u", "handler.py"]
