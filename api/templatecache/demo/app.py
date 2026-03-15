"""FastAPI demo app with /query, /stats, and frontend endpoints."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from templatecache.demo.savings_log import SavingsLog
from templatecache.main import TemplateCache

app = FastAPI(title="TemplateCache Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_cache = TemplateCache()
_log = SavingsLog()
_FRONTEND_PATH = Path(__file__).parent / "frontend.html"


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    prompt: str


@app.post("/query")
async def query_endpoint(request: QueryRequest) -> dict:
    """Process a query through the TemplateCache pipeline.

    Args:
        request: QueryRequest with a prompt field.

    Returns:
        Full pipeline response dict.
    """
    result = await _cache.query(request.prompt)
    _log.record(result)
    return result


@app.get("/stats")
async def stats_endpoint() -> dict:
    """Return aggregate savings statistics.

    Returns:
        Dict with total_requests, cache_hit_rate, average_savings_ratio,
        total_tokens_saved, slots_served_from_cache, slots_served_from_inference.
    """
    return _log.stats()


@app.get("/stats/history")
async def stats_history_endpoint() -> list:
    """Return per-request token savings history for graphing."""
    return _log.history()


@app.get("/", response_class=HTMLResponse)
async def frontend() -> str:
    """Serve the frontend HTML page.

    Returns:
        HTML content for the chat interface.
    """
    return _FRONTEND_PATH.read_text()

