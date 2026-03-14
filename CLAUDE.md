# CLAUDE.md — TemplateCache

## Project overview

TemplateCache is a modular LLM response caching system that sits between an
application and an LLM API. It reduces output token usage by 55–95% on
repetitive and mixed query traffic by reusing cached response structures and
only regenerating variable parts of a response.

The system combines dynamic template extraction from LLM outputs, slot-level
confidence scoring against historical fills, dependency-ordered partial
inference, gap detection for partial cache hits, and multi-query splitting
into a single drop-in caching primitive.

Runs 100% locally by default using sentence-transformers for embeddings and
Ollama for LLM generation. No API keys required.

---

## Architecture

The pipeline has three self-contained modules plus utility layers for
multi-query splitting and gap detection. Nothing outside these modules
should talk to Redis or the LLM API directly.
```
User query
    │
    ▼
split_multi_query()           — multiple distinct topics?
    │
    ├── yes → route each sub-question independently
    │
    └── no  → single query path
                  │
                  ▼
        IntentRouter.route()  — does a template exist for this query type?
              │
              ├── miss → full LLM generation → extract template → async write-back
              │
              └── hit → detect_query_gaps() — does the query ask for more than cached?
                            │
                            ├── no gaps → serve cached template directly
                            │
                            └── gaps found → SlotEngine.fill(gaps=gaps)
                                      │
                                      ├── confident slots → served from CacheStore
                                      ├── uncertain slots → targeted LLM generation
                                      └── supplement slots → targeted LLM for gaps
                                                │
                                                ▼
                                          Response stitcher → final response
                                                │
                                                ▼
                                          Async write-back → CacheStore
    │
    ▼
Combine sub-results (if multi-query) → final response
```

### Module responsibilities

**`modules/router.py` — IntentRouter**
Embeds the query, compares against intent centroid embeddings, returns an
intent ID and variant tag if similarity exceeds threshold. Answers one
question: does a reusable template exist?

**`modules/slot_engine.py` — SlotEngine**
Loads the template for an intent, checks slot confidence, fills uncertain
slots via targeted LLM calls in dependency order, creates supplement slots
for detected gaps, stitches all fills into the skeleton. Answers one
question: what does this template look like filled with this query's specifics?

**`modules/cache_store.py` — CacheStore**
Redis-backed persistence for templates, slot records, and intent centroids.
Three operations: get_template, get_slot_confidence, write_back. All other
modules talk to Redis exclusively through this class.

---

## File structure
```
templatecache/
├── main.py                  # TemplateCache entry point, query() and _query_single()
├── config.py                # All tunable constants, never hardcode elsewhere
├── modules/
│   ├── router.py            # IntentRouter
│   ├── slot_engine.py       # SlotEngine (includes supplement slot logic)
│   └── cache_store.py       # CacheStore
├── models/
│   ├── template.py          # ResponseTemplate dataclass
│   ├── slot.py              # SlotRecord dataclass
│   └── intent.py            # IntentCentroid dataclass
├── utils/
│   ├── embedder.py          # embed(), cosine_similarity(), batch_embed()
│   ├── extractor.py         # extract_template(), determine_variant(),
│   │                        # detect_query_gaps(), split_multi_query()
│   └── llm.py               # llm_call() — Ollama (local) or Gemini (remote)
├── demo/
│   ├── app.py               # FastAPI app, /query and /stats endpoints
│   ├── frontend.html         # Dark-themed UI with stitch visualization
│   └── savings_log.py       # SavingsLog, in-memory request log
├── tests/
│   ├── test_router.py
│   ├── test_slot_engine.py
│   └── test_cache_store.py
seed_cache.py                # Seeds 100 example query-response pairs into Redis
start.sh                     # Startup script, auto-detects Python and local mode
.env                         # Environment config (gitignored)
```

---

## Key constants (config.py)

| Constant | Default (local) | Default (remote) | Purpose |
|---|---|---|---|
| `USE_LOCAL_EMBEDDINGS` | true | false | Use sentence-transformers instead of Gemini embeddings |
| `USE_LOCAL_LLM` | true | false | Use Ollama instead of Gemini for generation |
| `OLLAMA_MODEL` | gemma3:4b | — | Local LLM model name |
| `OLLAMA_BASE_URL` | http://localhost:11434 | — | Ollama server URL |
| `LOCAL_EMBEDDING_MODEL` | all-MiniLM-L6-v2 | — | Local embedding model |
| `INTENT_SIMILARITY_THRESHOLD` | 0.55 | 0.90 | Cosine cutoff for intent match (lower for local models) |
| `SLOT_CONFIDENCE_THRESHOLD` | 0.50 | 0.85 | Similarity score to serve slot from cache |
| `UNCERTAIN_SLOT_FALLBACK_RATIO` | 0.5 | 0.5 | Fraction of uncertain slots that triggers fallback |
| `CONFIDENCE_DECAY_FACTOR` | 0.95 | 0.95 | Per-period decay multiplier for slot confidence |
| `CONFIDENCE_DECAY_DAYS` | 30 | 30 | Period length for decay calculation |
| `EMBEDDING_MODEL` | — | gemini-embedding-001 | Remote embedding model |
| `LLM_MODEL` | — | gemini-2.0-flash | Remote LLM model |
| `VARIANT_SHORT_MAX_TOKENS` | 80 | 80 | Response length ceiling for short variant |
| `VARIANT_DETAILED_MIN_TOKENS` | 200 | 200 | Response length floor for detailed variant |

---

## Data model quick reference

**ResponseTemplate**
Skeleton string with `[slot_name]` markers, ordered slot list, dependency
graph, variant tag, intent ID, hit count, created_at timestamp.

**SlotRecord**
Keyed by `slot:{slot_id}:{context_hash}` in Redis. Stores fill value, fill
embedding, similarity score, decay weight, created_at timestamp.

**IntentCentroid**
Keyed by `intent:{intent_id}` in Redis. Stores centroid embedding vector,
template ID, variant tag, query count.

---

## Redis key schema

| Key pattern | Contains |
|---|---|
| `template:{intent_id}` | Serialized ResponseTemplate |
| `slot:{slot_id}:{context_hash}` | Serialized SlotRecord |
| `intent:{intent_id}` | Serialized IntentCentroid |

No freeform key naming. All reads and writes go through CacheStore methods.

---

## Implementation rules Claude must follow

- All LLM calls go through `utils/llm.py:llm_call(prompt, max_tokens)` only.
  Never call Ollama or Gemini directly inside a module.
- All similarity comparisons use `embedder.cosine_similarity()` only.
  No inline dot products or numpy operations outside embedder.py.
- All config values imported from `config.py`. No hardcoded thresholds,
  model names, or Redis keys inside module code.
- Write-back must never block the response path. Always use
  `asyncio.create_task(cache_store.write_back(...))`.
- Every public method must have a docstring describing inputs, outputs,
  and side effects.
- Redis connection is instantiated once in CacheStore.__init__ and reused.
  No per-call connections.
- Embedding calls are cached in memory within a session in embedder.py.
  Do not re-embed identical strings.
- LLM prompts must be concise and direct. Minimize input tokens.
  Truncate context when passing existing responses to supplement prompts.

---

## Pipeline return contract

`TemplateCache.query(prompt)` always returns a dict with these fields:
```python
{
    "response": str,
    "cache_hit": bool,           # True if any sub-query matched cache
    "intent_id": str | None,     # Comma-separated IDs for multi-query
    "slots_from_cache": int,
    "slots_from_inference": int,
    "estimated_full_tokens": int,
    "actual_tokens_used": int,
    "savings_ratio": float,
    "stitch": {                  # Optional, present on cache hits
        "skeleton": str,
        "slot_fills": dict,
        "slot_sources": dict,
        "has_slots": bool,
        "gaps_detected": list | None,
        "multi_query": bool,     # True if query was split
        "sub_results": [         # Per-sub-question breakdown
            {"sub_query": str, "cache_hit": bool, "intent_id": str}
        ]
    }
}
```

Never return a partial dict. If any stage fails, catch the exception, log it,
and fall back to full generation with `cache_hit: False`.

---

## Slot filling rules

1. Load slot dependency order from the template before filling anything.
2. Fill slots sequentially in dependency order — never in parallel.
3. For each slot, run `cache_store.get_slot_confidence()` before any LLM call.
4. If confidence score exceeds `SLOT_CONFIDENCE_THRESHOLD`, use cached value.
5. If uncertain slot count divided by total slot count exceeds
   `UNCERTAIN_SLOT_FALLBACK_RATIO`, abort template filling and fall back to
   full generation.
6. When calling LLM for a slot, pass query context and all already-filled
   dependency slot values in the prompt via `_build_slot_prompt()`.
7. Never stitch until all slots are resolved.

---

## Gap detection rules

1. Split the query into aspects using commas, "and", and other conjunctions.
2. Embed each aspect and the cached response skeleton.
3. If an aspect's similarity to the skeleton is below 0.3, it's a gap.
4. Gaps become supplement slots appended to the template.
5. Each supplement slot gets a targeted LLM call: "Already answered: [truncated
   cached response]. Now answer ONLY this part: [gap]."
6. If the supplement LLM call fails, serve the cached part only with a
   `supplement_error` flag in the stitch info.

---

## Multi-query splitting rules

1. Split query on commas, question marks, and conjunctions.
2. Filter parts: must start with a question word and contain a specific topic
   (reject generic fragments like "give me examples").
3. Embed remaining parts and check pairwise cosine similarity.
4. If parts are semantically distinct (similarity < 0.45), treat as multi-query.
5. Route each sub-question through `_query_single()` independently.
6. Combine results: merge responses, aggregate token counts, track per-sub-query
   cache hit/miss status.
7. `cache_hit` is True if any sub-query hit the cache.

---

## Variant routing rules

| Signal | Variant |
|---|---|
| Query length < 8 words, no elaboration keywords | short |
| Query contains: explain, detail, describe, how does, walk me through | detailed |
| Query contains: list, what are, give me, options, compare | list |
| Response token count < `VARIANT_SHORT_MAX_TOKENS` | short |
| Response token count > `VARIANT_DETAILED_MIN_TOKENS` | detailed |

Variant is determined at centroid seeding time for cached templates and at
query time for routing. Both must agree or the detailed variant wins.

---

## Startup behaviour

On first run, if no intent centroids exist in Redis, run `seed_cache.py` to
populate 100 example query-response pairs across 90 intent types. Seeding
uses local embeddings (no API calls). The server auto-seeds with 5 minimal
examples if Redis is empty, but `seed_cache.py` provides much better coverage.

Start the server with `./start.sh` — it auto-detects the correct Python,
checks local mode settings, and launches uvicorn.

---

## Demo endpoints

`POST /query` — accepts `{"prompt": str}`, returns full pipeline response dict.

`GET /stats` — returns aggregate from SavingsLog:
- total_requests
- cache_hit_rate
- average_savings_ratio
- total_tokens_saved
- slots_served_from_cache
- slots_served_from_inference

`GET /` — serves the dark-themed frontend with query input, response display,
stitch visualization (shows skeleton, slot fills, gap detection, and
multi-query routing breakdown), and live stats.

---

## Testing expectations

Each module has a dedicated test file. Tests must cover:

**test_router.py** — above-threshold query returns intent ID, below-threshold
returns None, variant selection responds to query keywords correctly.

**test_slot_engine.py** — confident slots skip LLM call, uncertain slots
invoke LLM, fallback triggers when uncertain ratio exceeds threshold,
`_stitch()` replaces all markers with no leftovers.

**test_cache_store.py** — write and read round-trip for templates and slot
records, decay reduces score for records older than CONFIDENCE_DECAY_DAYS,
missing keys return None without raising.

Run all tests with `pytest tests/` from the project root.
Tests mock LLM and embedding calls — no Ollama or API needed to run them.

---

## Local-first setup

1. Install Ollama: `brew install ollama && brew services start ollama`
2. Pull a model: `ollama pull gemma3:4b`
3. Start Redis: `redis-server` or `brew services start redis`
4. Set `.env`: `USE_LOCAL_EMBEDDINGS=true` and `USE_LOCAL_LLM=true`
5. Seed cache: `python3 seed_cache.py`
6. Start server: `./start.sh`

No API keys needed. Everything runs on localhost.

---

## What this project is not

- Not a vector database. Redis is used as a lightweight store. Do not
  introduce Qdrant, Weaviate, or FAISS unless explicitly asked.
- Not a RAG pipeline. There is no document retrieval. Slots are filled from
  query context and historical fills only.
- Not a prompt caching layer. This operates entirely on the output side.
  Input/prefix caching is out of scope.
- Not a fine-tuning system. The LLM is never modified. The cache improves
  through accumulated slot confidence scores only.