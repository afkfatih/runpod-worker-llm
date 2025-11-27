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

# Override ENTRYPOINT to run our handler instead of vLLM server
ENTRYPOINT []
CMD ["/usr/bin/python3", "-u", "handler.py"]
