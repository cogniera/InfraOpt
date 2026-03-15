# CLAUDE.md — TemplateCache

## Project overview

TemplateCache is a modular LLM response caching system that sits between an
application and an LLM API. It reduces output token usage by 55–95% on
repetitive and mixed query traffic by reusing cached response structures and
only regenerating variable parts of a response.

The system combines dynamic template extraction from LLM outputs, slot-level
confidence scoring with cross-query transfer, confidence-weighted response
blending, gap detection with pattern learning and auto-slot promotion, and
multi-query splitting into a single drop-in caching primitive.

Runs 100% locally by default using sentence-transformers for embeddings and
Ollama for LLM generation. No API keys required.

---

## Architecture
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
              └── hit → detect_query_gaps() — does the query ask for more?
                            │
                            ├── no gaps → serve cached template directly
                            │
                            └── gaps found → SlotEngine.fill(gaps=gaps)
                                      │
                                      ├── confident slots (≥ SLOT_BLEND_THRESHOLD)
                                      │     → served from CacheStore directly
                                      │
                                      ├── blend zone slots (between thresholds)
                                      │     → weighted draw: cached vs fresh LLM
                                      │
                                      ├── uncertain slots (< SLOT_CONFIDENCE_THRESHOLD)
                                      │     → check cross-query transfer first
                                      │     → targeted LLM if transfer misses
                                      │
                                      └── supplement slots → targeted LLM for gaps
                                                │
                                                ▼
                                          Response stitcher
                                                │
                                                ▼
                                          GapLearner.store_gap()
                                          GapLearner.check_promotion()
                                                │
                                                ▼
                                          Async write-back → CacheStore
    │
    ▼
Combine sub-results (if multi-query) → final response
```

### Module responsibilities

**`modules/router.py` — IntentRouter**
Embeds the query, compares against intent centroid embeddings using two-phase
cluster search, updates centroid via online averaging on every hit. Returns
intent ID and variant tag if similarity exceeds threshold.

**`modules/slot_engine.py` — SlotEngine**
Loads template, runs three-state slot filling (confident / blend / uncertain),
attempts cross-query transfer before LLM calls, fills supplement slots for
gaps, stitches all fills into skeleton.

**`modules/cache_store.py` — CacheStore**
Redis-backed persistence for templates, slot records, intent centroids, and
gap event counters. All Redis access goes through this class exclusively.

**`modules/gap_learner.py` — GapLearner**
Classifies gap types, stores gap events in Redis, promotes recurring gap
types to permanent template slots after hitting GAP_PROMOTION_THRESHOLD.

---

## File structure
```
templatecache/
├── main.py                  # TemplateCache entry point, query() method
├── config.py                # All tunable constants, never hardcode elsewhere
├── modules/
│   ├── router.py            # IntentRouter
│   ├── slot_engine.py       # SlotEngine
│   ├── cache_store.py       # CacheStore
│   └── gap_learner.py       # GapLearner
├── models/
│   ├── template.py          # ResponseTemplate dataclass
│   ├── slot.py              # SlotRecord dataclass
│   └── intent.py            # IntentCentroid dataclass
├── utils/
│   ├── embedder.py          # embed(), cosine_similarity(), batch_embed()
│   ├── extractor.py         # extract_template(), determine_variant(),
│   │                        # detect_query_gaps(), split_multi_query(),
│   │                        # classify_slot()
│   └── llm.py               # llm_call() — Ollama or Gemini
├── demo/
│   ├── app.py               # FastAPI app, /query and /stats endpoints
│   ├── frontend.html        # Dark-themed UI with stitch visualization
│   └── savings_log.py       # SavingsLog, in-memory request log
├── tests/
│   ├── test_router.py
│   ├── test_slot_engine.py
│   ├── test_cache_store.py
│   └── test_gap_learner.py
├── seed_cache.py            # Seeds 100 example query-response pairs
├── start.sh                 # Startup script, auto-detects Python and local mode
└── .env                     # Environment config (gitignored)
```

---

## Key constants (config.py)

| Constant | Default (local) | Default (remote) | Purpose |
|---|---|---|---|
| `USE_LOCAL_EMBEDDINGS` | true | false | Use sentence-transformers |
| `USE_LOCAL_LLM` | true | false | Use Ollama for generation |
| `OLLAMA_MODEL` | gemma3:4b | — | Local LLM model |
| `OLLAMA_BASE_URL` | http://localhost:11434 | — | Ollama server URL |
| `LOCAL_EMBEDDING_MODEL` | all-MiniLM-L6-v2 | — | Local embedding model |
| `INTENT_SIMILARITY_THRESHOLD` | 0.55 | 0.90 | Cosine cutoff for intent match |
| `SLOT_CONFIDENCE_THRESHOLD` | 0.50 | 0.85 | Lower bound: uncertain below this |
| `SLOT_CONFIDENCE_THRESHOLDS` | see below | see below | Per-type confidence thresholds, overrides SLOT_CONFIDENCE_THRESHOLD per slot |
| `SLOT_BLEND_THRESHOLD` | 0.65 | 0.92 | Upper bound: confident above this |
| `SLOT_TRANSFER_PENALTY` | 0.15 | 0.15 | Confidence penalty for cross-template transfer |
| `SLOT_TRANSFER_ENABLED` | true | true | Enable cross-query slot transfer |
| `SLOT_BLEND_ENABLED` | true | true | Enable confidence-weighted blending |
| `GAP_PROMOTION_THRESHOLD` | 3 | 3 | Gap fires to trigger slot promotion |
| `GAP_LEARNING_ENABLED` | true | true | Enable gap pattern learning |
| `UNCERTAIN_SLOT_FALLBACK_RATIO` | 0.5 | 0.5 | Uncertain fraction triggering fallback |
| `CONFIDENCE_DECAY_FACTOR` | 0.95 | 0.95 | Per-period decay multiplier |
| `CONFIDENCE_DECAY_DAYS` | 30 | 30 | Decay period length |
| `EMBEDDING_MODEL` | — | gemini-embedding-001 | Remote embedding model |
| `LLM_MODEL` | — | gemini-2.0-flash | Remote LLM model |
| `VARIANT_SHORT_MAX_TOKENS` | 80 | 80 | Short variant ceiling |
| `VARIANT_DETAILED_MIN_TOKENS` | 200 | 200 | Detailed variant floor |

---

## Data model quick reference

**ResponseTemplate**
Skeleton string with `[slot_name]` markers, ordered slot list, slot dependency
graph, variant tag, intent ID, hit count, created_at timestamp. The dependency
graph maps each slot to the list of previously extracted slots that appear on
the same line — a slot depends on all earlier slots that co-occur on its line
in the skeleton.

**SlotRecord**
Keyed by `slot:{slot_id}:{context_hash}`. Fields: fill_value, fill_embedding,
similarity_score, decay_weight, created_at, slot_type.

`slot_type` is one of: CURRENCY, DATE, DURATION, NAMED_ENTITY, BOILERPLATE,
FREE_TEXT. Populated at extraction time by `classify_slot()` in extractor.py.

**IntentCentroid**
Keyed by `intent:{intent_id}`. Fields: centroid_embedding, template_id,
variant tag, query_count. Centroid embedding is updated via online averaging
on every cache hit.

---

## Redis key schema

| Key pattern | Contains |
|---|---|
| `template:{intent_id}` | Serialized ResponseTemplate |
| `slot:{slot_id}:{context_hash}` | Serialized SlotRecord |
| `intent:{intent_id}` | Serialized IntentCentroid |
| `gap:{template_id}:{gap_type}` | Gap event counter and aspect list |
| `promoted:{template_id}` | Set of gap types already promoted to slots |

No freeform key naming. All reads and writes go through CacheStore methods.

---

## Intent routing — domain and subdomain tiebreak

Implemented in `modules/cluster_router.py` as `_domain_tiebreak()` and
`_subdomain_tiebreak()`.

Routing uses a three-tier decision process:

**Tier 1 — Domain keyword matching** separates space from geography from
finance etc. When two or more candidate centroids score above
`INTENT_SIMILARITY_THRESHOLD` and the top two are within 0.05 cosine
similarity, domain keyword matching breaks the tie. If the top candidate's
domain has zero keyword matches but the query has a strong signal (1+ keyword)
for a different domain, the best candidate matching that domain is selected.

**Five domain keyword sets:**

| Domain | Keywords |
|---|---|
| space | planet, planets, solar, star, orbit, moon, nasa, galaxy, asteroid, comet, telescope, jupiter, saturn, mars, mercury, venus, neptune, uranus, solar system, dwarf planet, largest planet, smallest planet |
| geography | country, city, capital, continent, area, population, border, territory |
| finance | price, cost, dollar, market, stock, revenue, profit, budget |
| biology | cell, organism, species, gene, protein, evolution, ecosystem |
| history | war, century, empire, revolution, treaty, civilization |

Intent domains are resolved from the intent ID prefix (e.g. `space_` → space,
`comp_` → geography, `fin_` → finance) or the cluster label.

**Tier 2 — Subdomain keyword matching** separates planets from stars from
galaxies within the space domain (and countries from cities within geography).
When multiple candidates share the same domain, subdomain scoring picks the
best match. Each candidate's intent ID is scored against the query's best
subdomain keywords. The candidate with the highest subdomain score wins.

Subdomain keyword sets:

| Domain | Subdomain | Keywords |
|---|---|---|
| space | planets | planet, planets, solar system, jupiter, saturn, mars, mercury, venus, earth, neptune, uranus, orbit, dwarf planet |
| space | stars | star, stars, sun, supernova, hypergiant, red giant, white dwarf, neutron star, pulsar, uy scuti |
| space | galaxies | galaxy, milky way, andromeda, nebula, black hole, quasar |
| geography | countries | country, countries, nation, territory, border, capital |
| geography | cities | city, cities, urban, metropolitan, population |

**Cross-domain mismatch correction:** In `route()`, after the two-phase
cluster search returns a winner, the winner's domain is compared against
the query's domain (via `_get_query_domain()`). If they differ, cross-domain
rescue fires **unconditionally** — it does not require candidates to be
within 0.05 of each other. This is the primary defence against embedding
similarity pulling a query into the wrong domain (e.g. "largest planet"
matching `comp_largest_country` because "largest" dominates the embedding).

**Subdomain rescue scan:** `_subdomain_rescue_scan()` runs against ALL
centroids (not just those in the top-k clusters). It handles two cases:
1. **Cross-domain:** winner is in the wrong domain entirely. The scan
   finds the best candidate in the query's domain, filtered by subdomain
   if the query has subdomain keywords.
2. **Same-domain:** winner is in the right domain but wrong subdomain.
   The scan finds a candidate in the correct subdomain.

Candidates must score above `INTENT_SIMILARITY_THRESHOLD - 0.15` (rescue
floor). Word-boundary matching prevents "planet" from matching "exoplanet".

The rescue scan is O(n) over all centroids — keyword string matching plus
one cosine similarity call per candidate in the target domain.

**Tier 3 — Embedding similarity fallback** when subdomain scores are tied,
the candidate with the higher cosine similarity wins.

---

## Implementation rules Claude must follow

- All LLM calls go through `utils/llm.py:llm_call(prompt, max_tokens)` only.
- All similarity comparisons use `embedder.cosine_similarity()` only.
- All config values imported from `config.py`. No hardcoded values in modules.
- Write-back must never block the response path. Always use
  `asyncio.create_task(cache_store.write_back(...))`.
- Every public method must have a docstring.
- Redis connection instantiated once in CacheStore.__init__ and reused.
- Embedding calls cached in memory within a session in embedder.py.
- LLM prompts must be concise. Minimize input tokens.
- classify_slot() lives in utils/extractor.py only. Never duplicated.
- GapLearner is instantiated once in main.py and passed where needed.
- Never transfer slots across slot types.
- Blend zone only activates when SLOT_BLEND_ENABLED is true.
- Transfer only activates when SLOT_TRANSFER_ENABLED is true.
- Gap learning only activates when GAP_LEARNING_ENABLED is true.
- Slot confidence evaluation must use SLOT_CONFIDENCE_THRESHOLDS.get(slot_type,
  SLOT_CONFIDENCE_THRESHOLD) as the effective threshold. Never use the global
  SLOT_CONFIDENCE_THRESHOLD directly for per-slot decisions. The global value
  is a fallback only.
- `_clean_fill_value(value, slot_name)` must always receive `slot_name` as
  the second argument. This enables stripping the slot name and leaked slot
  name patterns from LLM-generated fill values. Omitting slot_name disables
  slot name contamination cleaning.

---

## Pipeline return contract

`TemplateCache.query(prompt)` always returns:
```python
{
    "response": str,
    "cache_hit": bool,
    "intent_id": str | None,
    "slots_from_cache": int,
    "slots_from_inference": int,
    "slots_from_transfer": int,
    "slots_from_blend": int,
    "estimated_full_tokens": int,
    "actual_tokens_used": int,
    "savings_ratio": float,
    "stitch": {
        "skeleton": str,
        "slot_fills": dict,
        "slot_sources": dict,
        "has_slots": bool,
        "gaps_detected": list[dict] | None,
        "slots_promoted": list[str],
        "blend_candidates": dict,
        "multi_query": bool,
        "sub_results": list[dict]
    }
}
```

Never return a partial dict. On any failure catch the exception, log it,
and fall back to full generation with `cache_hit: False`.

---

## Slot filling rules

1. Load slot dependency order from template before filling anything.
2. Fill slots sequentially in dependency order — never in parallel.
3. For each slot run get_slot_confidence() before any LLM call.
4. Three-state decision per slot:
   - score >= SLOT_BLEND_THRESHOLD → serve cached directly
   - SLOT_CONFIDENCE_THRESHOLD <= score < SLOT_BLEND_THRESHOLD → blend zone
   - score < SLOT_CONFIDENCE_THRESHOLD → uncertain
5. Blend zone: generate fresh fill via LLM, select between cached and fresh
   using confidence score as probability weight for cached value. Log both
   candidates and selected outcome in blend_candidates.
6. Uncertain slots: run _transfer_slot() first if SLOT_TRANSFER_ENABLED.
   Apply SLOT_TRANSFER_PENALTY to transfer candidate scores. Use transfer
   fill if penalised score exceeds SLOT_CONFIDENCE_THRESHOLD.
7. If transfer misses, call LLM via _build_slot_prompt() with query context
   and filled dependency values.
8. If uncertain slot count divided by total exceeds UNCERTAIN_SLOT_FALLBACK_RATIO
   (evaluated only when 3+ slots exist), abort and fall back to full generation.
9. Never stitch until all slots are resolved.
10. After stitching, if gaps were detected call GapLearner.store_gap() then
    GapLearner.check_promotion() for each gap. If promotion triggers, update
    the template in Redis immediately and log promoted slot names.
11. If query embedding similarity to template skeleton (first 200 chars) is
    below 0.4, build slot prompt without query context — use only the template
    skeleton and already-filled dependency slots. This prevents domain
    contamination when the router matches a semantically adjacent but
    domain-mismatched template.

---

## Gap detection rules

1. Split query into aspects on commas, "and", and other conjunctions.
2. Embed each aspect and the cached response skeleton.
3. Aspect similarity to skeleton below 0.3 → gap.
4. Gaps become supplement slots appended to the template.
5. Each supplement slot: targeted LLM call with truncated cached response
   as context, answering only the gap aspect.
6. If supplement LLM call fails, serve cached part only with supplement_error
   flag in stitch.
7. After serving, classify each gap type and pass to GapLearner.

---

## Gap type classification

Implemented in `gap_learner.py` as `classify_gap(aspect: str) -> str`.
Classification order (first match wins):

| Keywords in aspect | Gap type |
|---|---|
| when, date, year, month, time, recent, latest, update, current, now, today, ago | temporal |
| compar*, vs, versus, differ*, better, worse, than, contrast, alternative | comparison |
| how much/many/long/far/big/fast/often, cost, price, size, number, count, rate, percent | quantitative |
| why, caus*, reason, because, result, effect, impact, consequence, lead to | causal |
| how to, step(s), process, method, way to, procedure, guide | procedural |
| example, instance, sample, demonstrate, show me, such as, like what | example |
| anything else | elaboration |

---

## Gap promotion rules

1. Every gap event stored at `gap:{template_id}:{gap_type}` as counter + aspect list.
2. After every store, check_promotion() reads all gap counts for the template.
3. If any gap type count >= GAP_PROMOTION_THRESHOLD, promote it.
4. Promotion: load template, append `[{gap_type}_supplement_{n}]` slot to
   skeleton and slot order with no dependencies, save back to Redis.
5. Log promotion as `event_type: "slot_promoted"` in savings log.
6. Only promote each gap type once per template.
7. Do not promote if GAP_LEARNING_ENABLED is false.

---

## Answer extraction

When a cache hit matches a list-variant template and the query is a specific
factual question, the system extracts a single relevant item from the cached
list response rather than returning the full list.

Implemented in `utils/extractor.py` as `extract_specific_answer(query, response)`.
Called from `_query_single()` in `main.py` after template load and before gap
detection.

**Four gates — all must pass for extraction to run:**

1. Superlative signal — query contains at least one superlative keyword
   (largest, smallest, fastest, highest, lowest, closest, first, last, etc.)
2. Specific question starter — query contains a question starter pattern
   (what is the, what's the, which, who is the, tell me the, etc.)
3. Compound abort — query does not contain a compound conjunction
   (then, and also, as well as, plus, along with, and then)
   Compound queries fall through to gap detection which handles both parts.
4. List intent abort — query does not contain a list request signal
   (list, all of, every, enumerate, give me all, etc.)

If any gate fails, extract_specific_answer() returns None and the normal
gap detection path runs unchanged.

**Scoring**
After gates pass, each candidate list item is scored:
- +2 per criterion keyword found in item text or parenthetical metadata
- +3 for positional correctness (first item for closest/first, last for furthest/last)
- +3 for direct keyword match in item text

Item must reach ANSWER_EXTRACTION_MIN_SCORE (config.py, default 3.0) to be
returned. If no item reaches threshold, returns None and falls through to gap
detection.

**Config flags**
ANSWER_EXTRACTION_ENABLED — set False to disable entirely, all queries fall
through to gap detection.
ANSWER_EXTRACTION_MIN_SCORE — minimum score for an extracted item to be served.

**Stitch metadata fields added on extraction**
answer_extracted: True
extraction_criterion: the superlative keyword that matched
full_list_response: the original full list response before extraction

---

## Cross-query slot transfer rules

1. Only runs when SLOT_TRANSFER_ENABLED is true and slot is uncertain.
2. Scan all SlotRecord keys in Redis matching the same slot_type.
3. Compute cosine similarity between current query embedding and each
   candidate fill_embedding.
4. Subtract SLOT_TRANSFER_PENALTY from each score.
5. If best penalised score exceeds SLOT_CONFIDENCE_THRESHOLD, use that fill.
6. Log fill_source as "transfer" in slot_sources stitch metadata.
7. Never transfer across slot types.
8. Never transfer from a slot record with decay_weight below 0.3.

---

## Multi-query splitting rules

1. Split on commas, question marks, and conjunctions.
2. Filter: parts must start with a question word and contain a specific topic.
3. Embed remaining parts, check pairwise cosine similarity.
4. If similarity < 0.45, treat as multi-query.
5. Route each sub-question through _query_single() independently.
6. Combine: merge responses, aggregate tokens, track per-sub cache status.
7. cache_hit is True if any sub-query hit.

---

## Variant routing rules

| Signal | Variant |
|---|---|
| Query < 8 words, no elaboration keywords | short |
| explain, detail, describe, how does, walk me through | detailed |
| list, what are, give me, options, compare | list |
| Token count < VARIANT_SHORT_MAX_TOKENS | short |
| Token count > VARIANT_DETAILED_MIN_TOKENS | detailed |

---

## Slot type classification

Implemented in `utils/extractor.py` as `classify_slot(value: str) -> str`.

| Pattern | Type | Confidence threshold |
|---|---|---|
| Currency symbol + digits | currency | 0.85 |
| Date pattern (digits with / or -) | date | 0.82 |
| Range or qualified duration (5-7 business days, within 24 hours, 3 working days) | duration | 0.80 |
| Two or more capitalised words with optional lowercase particles (van, de, von, di) | named_entity | 0.80 |
| Plain day/time counts without business/working qualifier (30 days, 24 hours) | numeric | 0.78 |
| Length > 60 characters | boilerplate | 0.50 |
| Everything else | quoted_content | 0.65 |

**Known classification boundary:** Plain day counts without a business or working day
qualifier (e.g. "30 days", "24 hours") classify as numeric not duration. This is
intentional. The confidence thresholds for numeric (0.78) and duration (0.80) are
close enough that the behavioural difference is minimal. If a specific intent requires
plain day counts to be treated as duration, override the threshold explicitly in
`SLOT_CONFIDENCE_THRESHOLDS` in config.py rather than changing the classifier.

Called on every slot value at extraction time. Result stored on SlotRecord.

---

## Demo endpoints

`POST /query` — accepts `{"prompt": str}`, returns full pipeline dict.

`GET /stats` — returns from SavingsLog:
- total_requests
- cache_hit_rate
- average_savings_ratio
- total_tokens_saved
- slots_served_from_cache
- slots_served_from_inference
- slots_filled_via_transfer
- slots_filled_via_blend
- blend_cached_selected
- blend_fresh_selected
- gaps_recorded
- slots_promoted
- templates_evolved

`GET /` — serves dark-themed frontend with query input, response display,
stitch visualization, and live stats.

---

## Testing expectations

**test_router.py** — above-threshold returns intent ID, below returns None,
variant selection correct, centroid averaging updates embedding on hit.

**test_slot_engine.py** — confident slots skip LLM, uncertain slots invoke
LLM, blend zone generates both and selects one, fallback triggers at ratio
threshold, transfer fires before LLM on uncertain slots, transfer never
crosses slot types, _stitch() leaves no unreplaced markers.

**test_cache_store.py** — write/read round-trip for all record types, decay
reduces score for old records, gap counter increments correctly, missing keys
return None.

**test_gap_learner.py** — gap stored correctly by type, promotion fires at
threshold, promoted slot appears in subsequent template loads, promotion does
not fire below threshold, promotion does not fire when GAP_LEARNING_ENABLED
is false, gap types tracked independently per template.

Run all tests with `pytest tests/` from project root.
Tests mock LLM and embedding calls — no Ollama or API needed.

---

## Local setup

1. `brew install ollama && brew services start ollama`
2. `ollama pull gemma3:4b`
3. `redis-server` or `brew services start redis`
4. Set `.env`: USE_LOCAL_EMBEDDINGS=true, USE_LOCAL_LLM=true
5. `python3 seed_cache.py`
6. `./start.sh`

---

## Savings log fields (demo/savings_log.py)

Each request appended with:
- All fields from response contract
- slot_types dict (slot_name → slot_type for all slots in request)
- wrong_serves count (gaps triggered on cache-hit responses)
- blend_candidates dict
- gap_types list (gap type string for each gap detected)
- promoted_slots list

---

## What this project is not

- Not a vector database. Redis only. No Qdrant, Weaviate, or FAISS.
- Not a RAG pipeline. No document retrieval.
- Not a prompt caching layer. Output side only.
- Not a fine-tuning system. LLM never modified.