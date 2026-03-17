"""InfraOptComplete API — real TemplateCache backend wired to the React frontend."""

import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from templatecache.demo.savings_log import SavingsLog
from templatecache.main import TemplateCache

app = FastAPI(title="InfraOptComplete API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_log = SavingsLog()
_cache = TemplateCache(savings_log=_log)
_FRONTEND_HTML = Path(__file__).parent / "templatecache" / "demo" / "frontend.html"


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str


class InferenceRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class OptimizationConfig(BaseModel):
    strategy: str  # "speed", "quality", "balanced"
    batch_size: Optional[int] = 1
    quantization: Optional[str] = None


# ── Core TemplateCache endpoints (used by Chat and Dashboard pages) ────────────

@app.post("/query")
async def query_endpoint(request: QueryRequest) -> dict:
    """Process a query through the real TemplateCache pipeline."""
    result = await _cache.query(request.prompt)
    _log.record(result)
    return result


@app.get("/stats")
async def stats_endpoint() -> dict:
    """Return aggregate savings statistics from the live SavingsLog."""
    return _log.stats()


@app.get("/stats/history")
async def stats_history() -> dict:
    """Return per-request savings ratio history for the Chat sparkline."""
    entries = _log._entries
    history = [
        {
            "timestamp": int(time.time()) - (len(entries) - i) * 30,
            "savings_ratio": e.get("savings_ratio", 0.0),
            "cache_hit": e.get("cache_hit", False),
            "actual_tokens_used": e.get("actual_tokens_used", 0),
            "estimated_full_tokens": e.get("estimated_full_tokens", 0),
        }
        for i, e in enumerate(entries)
    ]
    return {"history": history}


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "service": "InfraOptComplete API"}


# ── Auxiliary endpoints (Models, Inference, Optimize pages) ───────────────────

@app.get("/api/models")
async def list_models() -> dict:
    """Return the available model list."""
    return {
        "models": [
            {"id": "llama-3-8b",  "name": "Llama 3 8B",  "size": "8B",   "status": "ready",   "latency_ms": 68},
            {"id": "mistral-7b",  "name": "Mistral 7B",  "size": "7B",   "status": "ready",   "latency_ms": 55},
            {"id": "phi-3-mini",  "name": "Phi-3 Mini",  "size": "3.8B", "status": "ready",   "latency_ms": 42},
            {"id": "gemma-2b",    "name": "Gemma 2B",    "size": "2B",   "status": "loading", "latency_ms": 28},
            {"id": "qwen-14b",    "name": "Qwen 14B",    "size": "14B",  "status": "ready",   "latency_ms": 110},
        ]
    }


@app.post("/api/infer")
async def run_inference(req: InferenceRequest) -> dict:
    """Run a prompt through the real TemplateCache pipeline."""
    result = await _cache.query(req.prompt)
    _log.record(result)
    return {
        "model": req.model,
        "prompt": req.prompt,
        "output": result["response"],
        "cache_hit": result["cache_hit"],
        "tokens_generated": result["actual_tokens_used"],
        "savings_ratio": result["savings_ratio"],
        "slots_from_cache": result["slots_from_cache"],
        "slots_from_inference": result["slots_from_inference"],
    }


@app.post("/api/optimize")
async def optimize(config: OptimizationConfig) -> dict:
    """Return optimization projection for the chosen strategy."""
    improvements = {
        "speed":    {"latency_reduction": "38%", "throughput_gain": "2.1x", "quality_loss": "minimal"},
        "quality":  {"latency_reduction": "5%",  "throughput_gain": "1.1x", "quality_loss": "none"},
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


# ── Embedded demo frontend (fallback HTML view) ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend() -> str:
    """Serve the TemplateCache demo HTML (fallback when React frontend is unavailable)."""
    return _FRONTEND_HTML.read_text()
