# RunPod Serverless Worker for GPT-OSS-20B

OpenAI Compatible Blazing-Fast LLM Endpoint powered by vLLM inference engine, optimized for OpenAI's GPT-OSS-20B model on RunPod Serverless.

## Model Specs

| Property | Value |
|----------|-------|
| Model | openai/gpt-oss-20b |
| Parameters | 21B (3.6B active - MoE) |
| VRAM Required | 16GB minimum |
| Max Context | 32K tokens |
| Quantization | MXFP4 (native) |
| License | Apache 2.0 |

## Quick Deploy on RunPod

### Option 1: Use Pre-built Docker Image

1. Go to [RunPod Console](https://runpod.io/console/serverless)
2. Create new Serverless Endpoint
3. Use Docker image: `your-dockerhub/runpod-gpt-oss-20b:latest`
4. Set environment variables (see below)
5. Select GPU: RTX 4090 (24GB) or better

### Option 2: Build Your Own Image

```bash
# Clone the repository
git clone https://github.com/your-username/runpod-worker-llm.git
cd runpod-worker-llm

# Build the Docker image
docker build -t your-dockerhub/runpod-gpt-oss-20b:latest .

# Push to Docker Hub
docker push your-dockerhub/runpod-gpt-oss-20b:latest
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | openai/gpt-oss-20b | HuggingFace model ID |
| `MAX_MODEL_LEN` | 32768 | Maximum context length |
| `GPU_MEMORY_UTILIZATION` | 0.95 | Fraction of GPU memory to use |
| `MAX_NUM_SEQS` | 256 | Max concurrent sequences |
| `TENSOR_PARALLEL_SIZE` | 1 | Number of GPUs for tensor parallelism |
| `MAX_CONCURRENCY` | 300 | Maximum concurrent requests |
| `HF_TOKEN` | - | HuggingFace token (if needed) |

## API Usage

### OpenAI Compatible Chat Completion

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Reasoning: medium"},
        {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ],
    temperature=1.0,
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

### Streaming Response

```python
stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Direct RunPod API Call

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

response = endpoint.run_sync({
    "input": {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 1.0,
        "max_tokens": 1024
    }
})

print(response)
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "messages": [
        {"role": "system", "content": "You are a helpful assistant. Reasoning: high"},
        {"role": "user", "content": "What is the meaning of life?"}
      ],
      "temperature": 1.0,
      "max_tokens": 2048
    }
  }'
```

## Reasoning Levels

GPT-OSS supports configurable reasoning effort via system prompt:

- **Low**: `"Reasoning: low"` - Fast responses for general dialogue
- **Medium**: `"Reasoning: medium"` - Balanced speed and detail
- **High**: `"Reasoning: high"` - Deep and detailed analysis

Example:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Reasoning: high"},
    {"role": "user", "content": "Solve this complex math problem..."}
]
```

## Recommended GPU Configurations

| GPU | VRAM | Recommended Config |
|-----|------|-------------------|
| RTX 4090 | 24GB | Single GPU, full context |
| A10G | 24GB | Single GPU, full context |
| L4 | 24GB | Single GPU, full context |
| A100-40GB | 40GB | Single GPU, extended batch |
| H100 | 80GB | Single GPU, max performance |

## Performance Tips

1. **Use Network Volume**: Attach RunPod network storage to cache model weights
2. **Set Active Workers**: Keep at least 1 active worker to avoid cold starts
3. **Optimize Context**: Use appropriate `MAX_MODEL_LEN` for your use case
4. **Batch Requests**: Group multiple requests when possible

## Local Testing

```bash
# Using docker-compose
docker-compose up --build

# Test with curl
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'
```

## Project Structure

```
runpod-worker-llm/
|-- Dockerfile           # Multi-stage build with vLLM
|-- handler.py           # RunPod serverless handler
|-- requirements.txt     # Python dependencies
|-- start.sh            # Startup script
|-- docker-compose.yml  # Local testing
|-- .env.example        # Environment template
|-- README.md           # This file
```

## Troubleshooting

### Out of Memory
- Reduce `MAX_MODEL_LEN` (try 16384 or 8192)
- Reduce `GPU_MEMORY_UTILIZATION` (try 0.90)
- Use a GPU with more VRAM

### Slow Cold Start
- Attach network storage with cached model
- Increase active workers count
- Use smaller batch sizes initially

### Model Not Loading
- Check HF_TOKEN if using gated models
- Verify CUDA compatibility
- Check GPU driver version

## License

Apache 2.0 - Same as GPT-OSS model

## Credits

- [OpenAI GPT-OSS](https://github.com/openai/gpt-oss)
- [vLLM](https://github.com/vllm-project/vllm)
- [RunPod](https://runpod.io)

