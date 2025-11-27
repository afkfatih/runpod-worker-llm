"""
RunPod Serverless Handler for GPT-OSS-20B
OpenAI Compatible API with vLLM Inference Engine
Optimized for fastest inference with full context support
"""

import os
import json
import time
import asyncio
from typing import Optional, List, Dict, Any, Union
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

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "256"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
DTYPE = os.getenv("DTYPE", "auto")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ENABLE_CHUNKED_PREFILL = os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "300"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
# CPU offload for using system RAM (in GB) - helps with limited VRAM
CPU_OFFLOAD_GB = float(os.getenv("CPU_OFFLOAD_GB", "0"))

# Global engine instance
engine: Optional[AsyncLLMEngine] = None
openai_serving_chat: Optional[OpenAIServingChat] = None


async def initialize_engine():
    """Initialize the vLLM engine with optimized settings for GPT-OSS-20B"""
    global engine, openai_serving_chat
    
    if engine is not None:
        return
    
    print(f"Initializing vLLM engine for {MODEL_NAME}...")
    print(f"Max context length: {MAX_MODEL_LEN}")
    print(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        disable_log_stats=os.getenv("DISABLE_LOG_STATS", "false").lower() == "true",
        enforce_eager=True,  # Disable CUDA graphs to avoid FlashAttention issues
        enable_prefix_caching=False,  # Disable for compatibility
        cpu_offload_gb=CPU_OFFLOAD_GB,  # Use system RAM for KV cache overflow
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
    
    print("vLLM engine initialized successfully!")


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

