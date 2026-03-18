# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

InfraOpt is an LLM inference optimization system called **TemplateCache**. It reduces redundant LLM calls by caching responses as parameterized templates with fillable slots. When a query matches a known intent, the system fills a cached template instead of calling the LLM.

## Architecture

**Request flow:**
1. Frontend (React/Vite, port 5173 dev / 80 prod) → Nginx → FastAPI (8000) + Express proxy (3000)
2. FastAPI receives query → **IntentRouter** embeds query and matches against stored intent centroids in Redis
3. On match: **SlotEngine** fills template slots using cached slot values or LLM inference
4. **GapLearner** detects unmet aspects in the query; promotes new slot types if gaps persist
5. On miss: call LLM, extract template from response, store in Redis for future use
6. **Redis** (6379) is the single persistence store for templates, slots, intent centroids, and metrics

**Key classes:**
- `TemplateCache` (`api/templatecache/main.py`) — top-level orchestrator
- `IntentRouter` (`modules/router.py`) — embedding-based intent matching
- `SlotEngine` (`modules/slot_engine.py`) — slot filling with confidence blending and decay
- `GapLearner` (`modules/gap_learner.py`) — detects coverage gaps, promotes new slots
- `CacheStore` (`modules/cache_store.py`) — Redis read/write for all persisted objects

**LLM backends** (switchable via `config.py` / `.env`):
- **Gemini 2.0 Flash** (default, `USE_LOCAL_LLM=false`) — higher similarity thresholds
- **Ollama gemma3:4b** (`USE_LOCAL_LLM=true`) — lower thresholds to compensate for weaker model

Similarity thresholds differ between backends — see `api/templatecache/config.py` for all tunable constants.

## Commands

### Full stack (Docker)
```bash
docker-compose up --build
```
On first start, `api/entrypoint.sh` waits for Redis, flushes it, seeds 300+ templates via `seed_cache.py` (takes a few minutes), then starts Uvicorn.

### Local development

**API (FastAPI):**
```bash
cd api
pip install -r requirements.txt
python -m uvicorn templatecache.demo.app:app --reload
```

**Backend proxy (Express):**
```bash
cd backend
npm install
node --watch server.js
```

**Frontend (React/Vite):**
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```
Vite dev server proxies `/query`, `/stats`, `/api/*`, `/health` to the API/backend automatically.

### Tests
```bash
cd api
pytest                                              # all tests
pytest templatecache/tests/test_router.py          # single file
pytest templatecache/tests/test_router.py::TestRouteAboveThreshold  # single class
pytest -v --cov=templatecache                      # verbose with coverage
```

Tests use `pytest-asyncio` for async functions. Test files are in `api/templatecache/tests/`.

### Seed cache manually
```bash
cd api
python seed_cache.py
```

## API Endpoints

**FastAPI (port 8000):**
- `POST /query` — main pipeline entry point
- `GET /stats` — aggregate savings (hit rate, tokens saved)
- `GET /stats/history` — per-request token savings history

**Express (port 3000):**
- Proxies `/api/*` and `/health` to FastAPI; `/node-health` for its own health check

## Important Notes

- **Redis must be running** before the API starts — it has a 30s startup timeout.
- **Similarity thresholds** are different for local vs. Gemini embeddings/LLM — changing `USE_LOCAL_LLM` or `USE_LOCAL_EMBEDDINGS` without adjusting thresholds will break routing quality.
- The `templatecache/` directory at the repo root appears to be a duplicate/symlink of `api/templatecache/` — make changes in `api/templatecache/`.
