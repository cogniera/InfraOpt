# InfraOpt

**Reduce LLM inference costs through intelligent template caching and slot filling.**

InfraOpt intercepts repeated or structurally similar queries, caches their response templates, and serves future answers by filling slots — cutting token usage by up to 90 % without sacrificing quality.

---

## How It Works

```
Query → Embed (OpenAI) → Intent Router (cosine sim)
                              │
              ┌───────────────┴───────────────┐
          Cache Hit                       Cache Miss
              │                               │
        Slot Engine                      Full LLM call
     (fill from cache +                       │
      targeted LLM)                   Extract template
              │                         & store in Redis
              ▼                               │
         Response                          Response
```

1. **Embed** — query is vectorised with OpenAI `text-embedding-3-small`.
2. **Route** — cosine similarity matches the query to a cached intent centroid.
3. **Hit path** — the SlotEngine fills template placeholders from Redis or short, targeted LLM calls.
4. **Miss path** — a full LLM generation runs, a reusable template is extracted and stored for next time.
5. **Gap learning** — recurring gaps are automatically promoted to permanent slots.

## Architecture

| Service | Stack | Port | Role |
|---------|-------|------|------|
| **api** | FastAPI · Python 3.12 | 8000 | TemplateCache engine (`/query`, `/stats`) |
| **backend** | Express · Node.js | 3000 | Proxy for `/api/*` stub routes |
| **frontend** | React 18 · Vite · Tailwind | 80 (nginx) | UI — chat, dashboard, pipeline animation |
| **redis** | Redis 7 Alpine | 6379 | Template, centroid & slot storage |

Nginx serves the SPA and reverse-proxies `/query` and `/stats` to the FastAPI service.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key

### Run

```bash
# 1. Add your key
echo "OPENAI_API_KEY=sk-..." > api/.env

# 2. Launch
docker compose up --build
```

The app is available at **http://localhost**.

### Local Development (no Docker)

```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn templatecache.demo.app:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev          # Vite dev server on :5173
```

## Project Structure

```
├── api/
│   ├── templatecache/
│   │   ├── main.py              # TemplateCache orchestrator
│   │   ├── config.py            # All tunable constants
│   │   ├── models/              # ResponseTemplate, IntentCentroid, SlotRecord
│   │   ├── modules/
│   │   │   ├── cache_store.py   # Redis abstraction
│   │   │   ├── router.py        # Intent routing via embeddings
│   │   │   ├── cluster_router.py# Fast two-step routing (50+ templates)
│   │   │   ├── slot_engine.py   # Template slot filling
│   │   │   └── gap_learner.py   # Gap detection & slot promotion
│   │   └── utils/
│   │       ├── embedder.py      # OpenAI embeddings + cosine similarity
│   │       ├── extractor.py     # Template extraction & variant detection
│   │       └── llm.py           # LLM call wrapper
│   ├── main.py                  # FastAPI app (health, inference, chat)
│   └── requirements.txt
├── backend/                     # Node.js proxy layer
├── frontend/
│   ├── src/
│   │   ├── pages/               # Landing, Chat, Dashboard, Inference, etc.
│   │   └── components/          # Sidebar, PipelineAnimation, StatCard
│   ├── nginx.conf               # Reverse proxy config
│   └── Dockerfile
└── docker-compose.yml
```

## Configuration

All tunables live in `api/templatecache/config.py` and can be overridden via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Model used for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for embeddings |
| `REDIS_HOST` | `localhost` | Redis host (Docker overrides to `redis`) |
| `INTENT_SIMILARITY_THRESHOLD` | `0.90` | Cosine sim threshold for cache hit |
| `SLOT_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence to serve a cached slot |
| `GAP_PROMOTION_THRESHOLD` | `3` | Gap occurrences before auto-promoting to slot |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Submit a query to the TemplateCache pipeline |
| `GET`  | `/stats` | Cache metrics — hit rate, savings, template count |
| `GET`  | `/health` | Health check |

## License

Private — Cogniera.
