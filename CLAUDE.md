# CLAUDE.md — TemplateCache

## Project overview

TemplateCache is a modular LLM response caching system that sits between an
application and an LLM API. It reduces output token usage by 55–75% on
repetitive and mixed query traffic by reusing cached response structures and
only regenerating variable parts of a response.

The system is novel in that it combines dynamic template extraction from LLM
outputs, slot-level confidence scoring against historical fills, and
dependency-ordered partial inference into a single drop-in caching primitive.
No existing open-source system does all three simultaneously.

---

## Architecture

The pipeline has three self-contained modules. Nothing outside these modules
should talk to Redis or the LLM API directly.
```
User query
    │
    ▼
IntentRouter.route()          — does a template exist for this query type?
    │
    ├── miss → full LLM generation → extract template → async write-back
    │
    └── hit → SlotEngine.fill()   — fill template with query specifics
                  │
                  ├── confident slots → served from CacheStore
                  │
                  └── uncertain slots → targeted LLM generation
                              │
                              ▼
                        Response stitcher → final response
                              │
                              ▼
                        Async write-back → CacheStore
```

### Module responsibilities

**`modules/router.py` — IntentRouter**
Embeds the query, compares against intent centroid embeddings, returns an
intent ID and variant tag if similarity exceeds threshold. Answers one
question: does a reusable template exist?

**`modules/slot_engine.py` — SlotEngine**
Loads the template for an intent, checks slot confidence, fills uncertain
slots via targeted LLM calls in dependency order, stitches fills into the
skeleton. Answers one question: what does this template look like filled with
this query's specifics?

**`modules/cache_store.py` — CacheStore**
Redis-backed persistence for templates, slot records, and intent centroids.
Three operations: get_template, get_slot_confidence, write_back. All other
modules talk to Redis exclusively through this class.

---

## File structure
```
templatecache/
├── main.py                  # TemplateCache entry point, query() method
├── config.py                # All tunable constants, never hardcode elsewhere
├── modules/
│   ├── router.py            # IntentRouter
│   ├── slot_engine.py       # SlotEngine
│   └── cache_store.py       # CacheStore
├── models/
│   ├── template.py          # ResponseTemplate dataclass
│   ├── slot.py              # SlotRecord dataclass
│   └── intent.py            # IntentCentroid dataclass
├── utils/
│   ├── embedder.py          # embed(), cosine_similarity(), batch_embed()
│   └── extractor.py         # extract_template(), determine_variant()
├── demo/
│   ├── app.py               # FastAPI app, /query and /stats endpoints
│   └── savings_log.py       # SavingsLog, in-memory request log
└── tests/
    ├── test_router.py
    ├── test_slot_engine.py
    └── test_cache_store.py
```

---

## Key constants (config.py)

| Constant | Default | Purpose |
|---|---|---|
| `INTENT_SIMILARITY_THRESHOLD` | 0.90 | Cosine distance cutoff for intent match |
| `SLOT_CONFIDENCE_THRESHOLD` | 0.85 | Similarity score to serve slot from cache |
| `UNCERTAIN_SLOT_FALLBACK_RATIO` | 0.5 | Fraction of uncertain slots that triggers full generation fallback |
| `CONFIDENCE_DECAY_FACTOR` | 0.95 | Per-period decay multiplier for slot confidence scores |
| `CONFIDENCE_DECAY_DAYS` | 30 | Period length for decay calculation |
| `EMBEDDING_MODEL` | text-embedding-3-small | Must match model used to build centroids |
| `LLM_MODEL` | gpt-4o-mini | Swap here only, never inside modules |
| `VARIANT_SHORT_MAX_TOKENS` | 80 | Response length ceiling for short variant |
| `VARIANT_DETAILED_MIN_TOKENS` | 200 | Response length floor for detailed variant |

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
  Never call the OpenAI client directly inside a module.
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

---

## Pipeline return contract

`TemplateCache.query(prompt)` always returns a dict with these fields:
```python
{
    "response": str,
    "cache_hit": bool,
    "intent_id": str | None,
    "slots_from_cache": int,
    "slots_from_inference": int,
    "estimated_full_tokens": int,
    "actual_tokens_used": int,
    "savings_ratio": float
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

On first run, if no intent centroids exist in Redis, `IntentRouter.seed_centroids()`
is called automatically with at least five example query-response pairs. These
must cover at least three distinct intent types and include at least one
example of each variant (short, detailed, list). Seeding blocks startup until
complete. Do not serve queries until seeding is finished.

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