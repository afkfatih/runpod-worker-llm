"""
RunPod Serverless Handler for GPT-OSS-20B
OpenAI Compatible API with vLLM Inference Engine
Optimized for fastest inference with full context support
Dynamic resource detection for optimal performance
"""

import os
import json
import time
import asyncio
from typing import Optional, List, Dict, Any, Union, Tuple
import runpod

# vLLM imports
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels

# Base configuration
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
DTYPE = os.getenv("DTYPE", "auto")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ENABLE_CHUNKED_PREFILL = os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)
# Allow manual override via env vars (0 = auto-detect)
MANUAL_MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "0"))
MANUAL_CPU_OFFLOAD_GB = float(os.getenv("CPU_OFFLOAD_GB", "-1"))
MANUAL_MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "0"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "100"))

# Global engine instance
engine: Optional[AsyncLLMEngine] = None
openai_serving_chat: Optional[OpenAIServingChat] = None


def detect_system_resources() -> Tuple[float, float, str]:
    """Detect GPU VRAM and system RAM dynamically"""
    import torch
    import psutil
    
    # Detect GPU VRAM
    gpu_vram_gb = 0.0
    gpu_name = "Unknown"
    if torch.cuda.is_available():
        gpu_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_vram_gb = gpu_vram_bytes / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
    
    # Detect system RAM
    system_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    return gpu_vram_gb, system_ram_gb, gpu_name


def calculate_optimal_settings(gpu_vram_gb: float, system_ram_gb: float) -> Dict[str, Any]:
    """
    Calculate optimal vLLM settings based on available resources.
    
    Based on vLLM documentation:
    - cpu_offload_gb: Offloads MODEL WEIGHTS to CPU (not KV cache)
    - swap_space: For KV cache swap when GPU memory is full
    - gpu_memory_utilization: Higher = more KV cache space
    - max_num_seqs: Lower = less concurrent requests, prevents preemption
    """
    # GPT-OSS-20B: ~16GB with MXFP4 quantization
    MODEL_SIZE_GB = 16.0
    
    # Available system RAM for offload (keep 8GB for system)
    available_ram = max(0, system_ram_gb - 8)
    
    # Determine optimal settings based on GPU VRAM
    if gpu_vram_gb >= 80:  # H100/H200 80GB, A100 80GB
        settings = {
            "max_model_len": 131072,  # Full 131K context
            "cpu_offload_gb": 0,       # No need
            "max_num_seqs": 256,       # High concurrency
            "gpu_memory_utilization": 0.95,
            "swap_space": 4,
        }
    elif gpu_vram_gb >= 48:  # A6000, L40S, A40
        settings = {
            "max_model_len": 131072,
            "cpu_offload_gb": 0,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.95,
            "swap_space": 8,
        }
    elif gpu_vram_gb >= 24:  # RTX 4090, A5000, L4
        settings = {
            "max_model_len": 65536,    # 64K context
            "cpu_offload_gb": 0,       # Model fits
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.95,
            "swap_space": 16,          # KV cache swap
        }
    elif gpu_vram_gb >= 20:  # RTX A4500, A4000 (20GB)
        # Model is 16GB, only 4GB left for KV cache
        # Need CPU offload for model weights if we want more context
        settings = {
            "max_model_len": 131072 if available_ram >= 24 else 32768,
            "cpu_offload_gb": min(8, available_ram * 0.2),  # Offload some model weights
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.98,
            "swap_space": min(32, available_ram * 0.4),  # KV cache swap to RAM
        }
    elif gpu_vram_gb >= 16:  # RTX 4080/3090 (16GB)
        # Very tight! Model barely fits
        settings = {
            "max_model_len": 32768,
            "cpu_offload_gb": min(8, available_ram * 0.2),
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.98,
            "swap_space": min(48, available_ram * 0.5),
        }
    else:
        # Insufficient VRAM - heavy offload required
        settings = {
            "max_model_len": 16384,
            "cpu_offload_gb": min(16, available_ram * 0.3),
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.98,
            "swap_space": min(48, available_ram * 0.5),
        }
    
    return settings


async def initialize_engine():
    """Initialize the vLLM engine with optimized settings for GPT-OSS-20B"""
    global engine, openai_serving_chat
    
    if engine is not None:
        return
    
    # Detect system resources
    gpu_vram_gb, system_ram_gb, gpu_name = detect_system_resources()
    print(f"=== System Resources Detected ===")
    print(f"GPU: {gpu_name}")
    print(f"GPU VRAM: {gpu_vram_gb:.1f} GB")
    print(f"System RAM: {system_ram_gb:.1f} GB")
    
    # Calculate optimal settings
    optimal = calculate_optimal_settings(gpu_vram_gb, system_ram_gb)
    
    # Allow manual override via environment variables
    max_model_len = MANUAL_MAX_MODEL_LEN if MANUAL_MAX_MODEL_LEN > 0 else optimal["max_model_len"]
    cpu_offload_gb = MANUAL_CPU_OFFLOAD_GB if MANUAL_CPU_OFFLOAD_GB >= 0 else optimal["cpu_offload_gb"]
    max_num_seqs = MANUAL_MAX_NUM_SEQS if MANUAL_MAX_NUM_SEQS > 0 else optimal["max_num_seqs"]
    gpu_mem_util = optimal["gpu_memory_utilization"]
    swap_space = optimal["swap_space"]
    
    print(f"\n=== Optimized Configuration ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Max context length: {max_model_len:,} tokens")
    print(f"CPU offload: {cpu_offload_gb:.1f} GB")
    print(f"Max concurrent sequences: {max_num_seqs}")
    print(f"GPU memory utilization: {gpu_mem_util:.0%}")
    print(f"Swap space: {swap_space:.1f} GB")
    
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=DTYPE,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        disable_log_stats=os.getenv("DISABLE_LOG_STATS", "false").lower() == "true",
        enforce_eager=True,
        enable_prefix_caching=False,
        cpu_offload_gb=cpu_offload_gb,
        swap_space=swap_space,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Initialize OpenAI serving chat
    model_config = await engine.get_model_config()
    openai_serving_chat = OpenAIServingChat(
        engine=engine,
        model_config=model_config,
        served_model_names=[MODEL_NAME],
        response_role="assistant",
    )
    
    print("\nvLLM engine initialized successfully!")


def create_sampling_params(request: Dict[str, Any]) -> SamplingParams:
    """Create sampling parameters from request"""
    return SamplingParams(
        temperature=request.get("temperature", 1.0),
        top_p=request.get("top_p", 1.0),
        top_k=request.get("top_k", -1),
        max_tokens=request.get("max_tokens", 2048),
        stop=request.get("stop", None),
        presence_penalty=request.get("presence_penalty", 0.0),
        frequency_penalty=request.get("frequency_penalty", 0.0),
        repetition_penalty=request.get("repetition_penalty", 1.0),
        n=request.get("n", 1),
        best_of=request.get("best_of", None),
        use_beam_search=request.get("use_beam_search", False),
        skip_special_tokens=request.get("skip_special_tokens", True),
        ignore_eos=request.get("ignore_eos", False),
        seed=request.get("seed", None),
    )


async def handle_chat_completion(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle chat completion request (OpenAI compatible)"""
    global engine, openai_serving_chat
    
    messages = request.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}
    
    model = request.get("model", MODEL_NAME)
    stream = request.get("stream", False)
    
    # Build the chat completion request
    chat_request = ChatCompletionRequest(
        model=model,
        messages=messages,
        temperature=request.get("temperature", 1.0),
        top_p=request.get("top_p", 1.0),
        max_tokens=request.get("max_tokens", 2048),
        stream=stream,
        stop=request.get("stop"),
        presence_penalty=request.get("presence_penalty", 0.0),
        frequency_penalty=request.get("frequency_penalty", 0.0),
        n=request.get("n", 1),
        seed=request.get("seed"),
    )
    
    try:
        if stream:
            # Streaming response
            generator = await openai_serving_chat.create_chat_completion(chat_request)
            chunks = []
            async for chunk in generator:
                if hasattr(chunk, 'model_dump'):
                    chunks.append(chunk.model_dump())
                else:
                    chunks.append(chunk)
            return {"chunks": chunks, "stream": True}
        else:
            # Non-streaming response
            response = await openai_serving_chat.create_chat_completion(chat_request)
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            return response
    except Exception as e:
        return {"error": str(e)}


async def handle_completion(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle text completion request"""
    global engine
    
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}
    
    sampling_params = create_sampling_params(request)
    request_id = f"cmpl-{int(time.time() * 1000)}"
    
    try:
        results_generator = engine.generate(prompt, sampling_params, request_id)
        final_result = None
        async for result in results_generator:
            final_result = result
        
        if final_result is None:
            return {"error": "No result generated"}
        
        # Format response
        choices = []
        for i, output in enumerate(final_result.outputs):
            choices.append({
                "index": i,
                "text": output.text,
                "finish_reason": output.finish_reason,
            })
        
        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(final_result.prompt_token_ids),
                "completion_tokens": sum(len(o.token_ids) for o in final_result.outputs),
                "total_tokens": len(final_result.prompt_token_ids) + sum(len(o.token_ids) for o in final_result.outputs),
            }
        }
    except Exception as e:
        return {"error": str(e)}


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function
    Supports both OpenAI chat completion and text completion formats
    """
    job_input = job.get("input", {})
    
    # Initialize engine if needed
    await initialize_engine()
    
    # Determine request type
    if "messages" in job_input:
        # Chat completion (OpenAI format)
        return await handle_chat_completion(job_input)
    elif "prompt" in job_input:
        # Text completion
        return await handle_completion(job_input)
    else:
        return {"error": "Invalid request. Provide 'messages' for chat or 'prompt' for completion."}


# RunPod serverless start
if __name__ == "__main__":
    print("Starting RunPod Serverless Worker for GPT-OSS-20B")
    print(f"Model: {MODEL_NAME}")
    print(f"Max context: {MAX_MODEL_LEN}")
    
    # Start the serverless handler
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda x: MAX_CONCURRENCY,
    })

