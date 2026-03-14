from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import random

app = FastAPI(title="InferOpt API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class ChatMessage(BaseModel):
    role: str      # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = "llama-3-8b"

class OptimizationConfig(BaseModel):
    strategy: str  # "speed", "quality", "balanced"
    batch_size: Optional[int] = 1
    quantization: Optional[str] = None  # "int8", "fp16", None

@app.get("/health")
async def health():
    return {"status": "ok", "service": "inferOpt API"}

@app.get("/api/stats")
async def get_stats():
    return {
        "total_requests": random.randint(1000, 9999),
        "avg_latency_ms": round(random.uniform(45, 120), 1),
        "throughput_rps": round(random.uniform(80, 200), 1),
        "gpu_utilization": round(random.uniform(60, 95), 1),
        "memory_used_gb": round(random.uniform(4, 14), 2),
        "uptime_hours": round(random.uniform(10, 500), 1),
    }

@app.get("/api/models")
async def list_models():
    return {
        "models": [
            {"id": "llama-3-8b", "name": "Llama 3 8B", "size": "8B", "status": "ready", "latency_ms": 68},
            {"id": "mistral-7b", "name": "Mistral 7B", "size": "7B", "status": "ready", "latency_ms": 55},
            {"id": "phi-3-mini", "name": "Phi-3 Mini", "size": "3.8B", "status": "ready", "latency_ms": 42},
            {"id": "gemma-2b", "name": "Gemma 2B", "size": "2B", "status": "loading", "latency_ms": 28},
            {"id": "qwen-14b", "name": "Qwen 14B", "size": "14B", "status": "ready", "latency_ms": 110},
        ]
    }

@app.post("/api/infer")
async def run_inference(req: InferenceRequest):
    start = time.time()
    # Simulate inference delay
    time.sleep(random.uniform(0.1, 0.4))
    elapsed = round((time.time() - start) * 1000, 1)

    return {
        "model": req.model,
        "prompt": req.prompt,
        "output": f"[Simulated output for '{req.prompt[:40]}...' using {req.model}]",
        "tokens_generated": random.randint(50, req.max_tokens),
        "latency_ms": elapsed,
        "tokens_per_second": round(random.uniform(30, 120), 1),
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    import asyncio
    start = time.time()
    await asyncio.sleep(random.uniform(0.4, 1.2))
    elapsed = round((time.time() - start) * 1000, 1)

    last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    reply = (
        f"This is a simulated response from **{req.model}**.\n\n"
        f"You asked: *\"{last[:80]}{'...' if len(last) > 80 else ''}\"*\n\n"
        "In a real deployment this would be the model's generated output. "
        "Connect your preferred inference backend to replace this stub."
    )
    return {
        "role": "assistant",
        "content": reply,
        "model": req.model,
        "tokens": random.randint(40, 220),
        "latency_ms": elapsed,
    }

@app.post("/api/optimize")
async def optimize(config: OptimizationConfig):
    improvements = {
        "speed": {"latency_reduction": "38%", "throughput_gain": "2.1x", "quality_loss": "minimal"},
        "quality": {"latency_reduction": "5%", "throughput_gain": "1.1x", "quality_loss": "none"},
        "balanced": {"latency_reduction": "22%", "throughput_gain": "1.6x", "quality_loss": "negligible"},
    }
    result = improvements.get(config.strategy, improvements["balanced"])
    return {
        "strategy": config.strategy,
        "batch_size": config.batch_size,
        "quantization": config.quantization,
        "projected_improvements": result,
        "status": "optimization applied",
    }

@app.get("/api/latency-history")
async def latency_history():
    now = int(time.time())
    points = []
    base = 80
    for i in range(20):
        base += random.uniform(-10, 12)
        base = max(20, min(200, base))
        points.append({
            "timestamp": now - (20 - i) * 30,
            "latency_ms": round(base, 1),
            "throughput": round(random.uniform(80, 180), 1),
        })
    return {"history": points}
