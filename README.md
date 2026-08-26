# RunPod Serverless Worker — GPT-OSS-20B

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/engine-vLLM-orange.svg)](https://github.com/vllm-project/vllm)

An OpenAI-compatible, serverless LLM endpoint for [RunPod](https://runpod.io), powered by the
[vLLM](https://github.com/vllm-project/vllm) inference engine and tuned for OpenAI's
[GPT-OSS-20B](https://github.com/openai/gpt-oss).

**What makes this different from a plain vLLM container:** the worker probes the GPU and system
RAM at startup and picks `max_model_len`, `cpu_offload_gb`, `max_num_seqs`, `gpu_memory_utilization`
and `swap_space` for whatever hardware RunPod happens to schedule it on — so the same image runs
on a 16 GB card and on an H100 without a config change. See
[Automatic resource detection](#automatic-resource-detection).

## Model Specs

| Property | Value |
|----------|-------|
| Model | `openai/gpt-oss-20b` |
| Parameters | 21B total, 3.6B active (MoE) |
| VRAM required | 16 GB minimum |
| Max context | up to 131K tokens (hardware dependent — see table below) |
| Quantization | MXFP4 (native) |
| License | Apache 2.0 |

## Quick Deploy on RunPod

No pre-built image is published — build and push your own, to any registry you already use.

```bash
git clone https://github.com/afkfatih/runpod-worker-llm.git
cd runpod-worker-llm

docker build -t <your-registry>/runpod-worker-llm:latest .
docker push <your-registry>/runpod-worker-llm:latest
```

Then:

1. Open the [RunPod Serverless console](https://runpod.io/console/serverless)
2. Create a new Serverless Endpoint
3. Container image: the tag you just pushed
4. Select a GPU — RTX 4090 (24 GB) or better is recommended
5. Optionally set the environment variables below (all have sane defaults)

> The image must be pullable by RunPod — either public, or added to your RunPod
> account as a private registry credential.

## Automatic resource detection

Every environment variable below defaults to auto-detection. On startup the worker reads the GPU's
total VRAM and the system RAM, then applies this profile:

| Detected VRAM | Example GPUs | Max context | CPU offload | Max sequences |
|---|---|---|---|---|
| ≥ 80 GB | H100, H200, A100-80 | 131,072 | none | 256 |
| ≥ 48 GB | A6000, L40S, A40 | 131,072 | none | 128 |
| ≥ 24 GB | RTX 4090, A5000, L4 | 65,536 | none | 64 |
| ≥ 20 GB | RTX A4500, A4000 | 131,072 if ≥ 24 GB RAM free, else 32,768 | up to 8 GB | 32 |
| ≥ 16 GB | RTX 4080 | 32,768 | up to 8 GB | 16 |
| < 16 GB | — | 16,384 | up to 16 GB | 8 |

Below 24 GB the worker offloads model weights to system RAM and swaps KV cache, so a 20 GB card can
still serve the full 131K context when enough host RAM is available.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `openai/gpt-oss-20b` | HuggingFace model ID |
| `MAX_MODEL_LEN` | `0` | Context length. `0` = auto-detect |
| `MAX_NUM_SEQS` | `0` | Max concurrent sequences. `0` = auto-detect |
| `CPU_OFFLOAD_GB` | `-1` | GB of model weights to offload to RAM. `-1` = auto-detect |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `MAX_CONCURRENCY` | `100` | Max concurrent requests per worker |
| `DTYPE` | `auto` | Model dtype |
| `TRUST_REMOTE_CODE` | `true` | Allow custom modeling code from the Hub |
| `ENABLE_CHUNKED_PREFILL` | `true` | Chunked prefill for long prompts |
| `DISABLE_LOG_STATS` | `false` | Silence vLLM throughput logging |
| `HF_TOKEN` | — | HuggingFace token, for gated models |

`gpu_memory_utilization` and `swap_space` are always derived from detected hardware and have no
environment override.

## API Usage

### OpenAI-compatible chat completion

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

### Direct RunPod API call

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

### Text completion

Send `prompt` instead of `messages`:

```python
response = endpoint.run_sync({
    "input": {"prompt": "Once upon a time", "max_tokens": 256}
})
```

### cURL

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

### A note on streaming

Passing `"stream": true` is accepted, but the handler collects every chunk and returns them as a
single `{"chunks": [...]}` payload — it is not incremental delivery over the wire. Token-by-token
streaming requires a RunPod generator handler; see [Known limitations](#known-limitations).

## Reasoning Levels

GPT-OSS takes its reasoning effort from the system prompt:

- `"Reasoning: low"` — fast responses for general dialogue
- `"Reasoning: medium"` — balanced speed and detail
- `"Reasoning: high"` — deep and detailed analysis

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Reasoning: high"},
    {"role": "user", "content": "Solve this complex math problem..."}
]
```

## Performance Tips

1. **Attach a network volume** so model weights are cached between cold starts — this is the single
   biggest latency win on serverless.
2. **Keep at least one active worker** if you care about p99 latency.
3. **Pin `MAX_MODEL_LEN`** to what you actually need; auto-detection is generous and KV cache scales
   with it.

## Local Testing

```bash
docker-compose up --build

curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'
```

## Project Structure

```
runpod-worker-llm/
├── Dockerfile           # Based on vllm/vllm-openai:gptoss
├── handler.py           # RunPod serverless handler + resource detection
├── requirements.txt     # Python dependencies
├── start.sh             # Startup script
├── docker-compose.yml   # Local testing
└── .env.example         # Environment template
```

## Known limitations

- **vLLM V0 engine.** The image pins `VLLM_USE_V1=0` and `enforce_eager=True` to work around a
  FlashAttention 3 requirement on non-Hopper GPUs. V0 is deprecated upstream; migrating to V1 is the
  main open piece of work.
- **Streaming is buffered,** not incremental — see the note above.
- **`enable_prefix_caching` is off,** which costs throughput on shared-prefix workloads.

## Troubleshooting

**Out of memory** — lower `MAX_MODEL_LEN` (try 16384 or 8192), or move to a GPU with more VRAM.

**Slow cold start** — attach network storage with the model cached, and raise the active worker
count.

**Model not loading** — check `HF_TOKEN` for gated models, and verify CUDA/driver compatibility.

## License

[Apache 2.0](LICENSE) — same as the GPT-OSS model.

## Credits

- [OpenAI GPT-OSS](https://github.com/openai/gpt-oss)
- [vLLM](https://github.com/vllm-project/vllm)
- [RunPod](https://runpod.io)
