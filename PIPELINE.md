# TemplateCache — Pipeline Reference

## Data Models

Three dataclasses define everything stored in Redis:

**ResponseTemplate** — A skeleton string with `[slot_name]` markers, an
ordered slot list, a dependency graph between slots, and a variant tag
(short/detailed/list).

**SlotRecord** — A cached fill value for a specific slot in a specific
context. Includes the fill embedding, a confidence score, a time-decay
weight, and a classified type (numeric, currency, duration, named_entity,
etc.).

**IntentCentroid** — An average embedding vector representing a query
type. Points to its template and tracks how many queries have contributed
to the centroid.

---

## The Pipeline

### Phase 0 — Startup seeding

On first run, if Redis is empty, `seed_centroids()` blocks startup and
processes 300+ example query-response pairs. For each pair: generate a
response via LLM, extract a template skeleton with slot markers, embed
the query to create a centroid, and write everything to Redis. The
ClusterRouter then groups all centroids into ~24 k-means clusters.

---

### Phase 1 — Multi-query split

`split_multi_query()` checks if the prompt contains multiple topics
(e.g. "what is Python and who created it"). If so, it splits into
sub-questions that are each processed independently through
`_query_single()`, then combined. Pronoun references in sub-questions
(e.g. "how does it work") are resolved by injecting the topic from the
original query before routing.

---

### Phase 2 — Intent routing

For each sub-query, routing answers: "Do we have a template for this?"
```
query → embed() → compare against centroid embeddings → intent_id + variant
```

**ClusterRouter** (604 centroids, 24 clusters):
1. Find the top-2 closest clusters by cosine similarity to query embedding
2. Search only centroids within those clusters
3. If winner's domain ≠ query's domain → cross-domain rescue scans ALL
   centroids in the correct domain unconditionally
4. If winner's domain = query's domain but wrong subdomain → subdomain
   rescue finds the correct subdomain match within the domain
5. Falls back to flat O(n) scan if clustering was not built

**Domain keyword sets** used for domain and subdomain detection:
- space: planet, solar, star, orbit, moon, galaxy, asteroid, jupiter,
  saturn, mars, mercury, venus, neptune, uranus, largest planet, etc.
- geography: country, city, capital, continent, area, population, border
- finance: price, cost, dollar, market, stock, revenue, profit
- biology: cell, organism, species, gene, protein, evolution
- history: war, century, empire, revolution, treaty, civilization

Subdomains (e.g. planets vs stars within space) provide a second tier
of disambiguation when two candidates share the same domain.

**IntentRouter** is the simpler fallback — flat scan over all centroids,
used during seeding before the cluster index is built.

**Thresholds vary by deployment mode:**

| Threshold | Local mode | Remote/production |
|---|---|---|
| `INTENT_SIMILARITY_THRESHOLD` | 0.55 | 0.90 |
| `SLOT_CONFIDENCE_THRESHOLD` | 0.50 | 0.85 |
| `SLOT_BLEND_THRESHOLD` | 0.65 | 0.92 |

Local mode is active when `USE_LOCAL_EMBEDDINGS=True` in `.env`. Remote
mode uses Gemini embeddings and higher thresholds because the embedding
space is more discriminative. Both sets are defined in `config.py` and
selected at runtime via environment variables.

---

### Phase 3a — Cache hit path

Template exists. Now fill it.

Example skeleton:
```
"[planet_name] is the [superlative] planet, with [fact_detail]."
```

**Templateable check** — if `template.templateable` is False, the
`raw_response` field is served directly with savings_ratio 1.0 and
zero LLM calls. Gap detection, slot filling, and answer extraction
are all skipped. This path handles code-heavy responses that were
flagged as non-templateable at extraction time.

**Pre-fill: Gap detection**
`detect_query_gaps()` runs against the template skeleton before slot
filling begins. Detected gaps are passed into `SlotEngine.fill()` as
the `gaps=` parameter so supplement slots can be created and filled
during the same stitch pass as regular slots.

**SlotEngine.fill()** processes slots in dependency order:

1. Load dependency graph — if `fact_detail` depends on `planet_name`,
   fill `planet_name` first
2. For each slot, check `CacheStore.get_slot_confidence()`:
   - **Confident** (score ≥ blend threshold) → serve directly from
     cache, zero tokens
   - **Blend zone** (score between confidence threshold and blend
     threshold) → call LLM for a fresh value, then probabilistically
     pick cached vs fresh based on confidence weighting
   - **Uncertain** (score < confidence threshold) → try slot transfer
     first (reuse a fill from a different query of the same slot type,
     with a 0.15 penalty applied to the score), then fall back to a
     targeted LLM call for just that one slot
3. Fallback ratio check — if more than 50% of slots are uncertain and
   there are 3 or more slots total, abort template filling entirely and
   fall back to full generation
4. Query context sanitisation — if the query embedding similarity to
   the template skeleton is below 0.4, the slot prompt is built WITHOUT
   the query text to prevent domain contamination
5. Type-specific formatting — numeric slots get "reply with
   human-readable numbers", currency slots get "reply with formatted
   currency", duration slots get "reply with human-readable duration"
6. Clean fill values — strip leaked slot names, bracket markers, and
   surrounding quotes via `_clean_fill_value(value, slot_name)`
7. Stitch — replace all `[slot_name]` markers with fill values. Safety
   regex removes any remaining unreplaced markers.

**Step 7.5 — Answer extraction**
If template variant is `"list"` and `ANSWER_EXTRACTION_ENABLED` is True:

- **Single factual query** (superlative signal + question starter, no
  compound conjunction, no list signal) → `extract_specific_answer()`
  scores each list item against the criterion keyword and returns the
  best match if score ≥ `ANSWER_EXTRACTION_MIN_SCORE`. Returned
  directly, gap learning still runs but gap detection is skipped.
- **Compound query** (superlative + list signal + conjunction) →
  `_format_compound_list_response()` extracts the specific answer and
  prepends it to the full list response. Returned directly.
- If extraction returns None, falls through to gap learning normally.

**Post-stitch: Gap learning**
`GapLearner` classifies each detected gap (temporal, comparison,
quantitative, causal, procedural, example), stores the count in Redis
at `gap:{template_id}:{gap_type}`, and promotes recurring gap types to
permanent template slots when count reaches `GAP_PROMOTION_THRESHOLD`
(default 3).

---

### Phase 3b — Cache miss path

No template exists:

1. Full LLM generation (300+ tokens)
2. `extract_template()` — identify variable parts, create skeleton with
   slot markers, classify slot types, check templateable flag (responses
   more than 60% code are not templated)
3. `determine_variant()` — classify as short/detailed/list based on
   query keywords and response length
4. Create `ResponseTemplate` and `IntentCentroid`
5. Async write-back to Redis — never blocks the response path

Next identical or similar query will hit the cache.

---

### Phase 4 — Response assembly

Every response, hit or miss, returns:
```python
{
    "response": "Jupiter is the largest planet...",
    "cache_hit": True,
    "intent_id": "space_solar_system",
    "slots_from_cache": 4,        # zero token cost
    "slots_from_inference": 2,    # targeted LLM calls
    "slots_from_transfer": 1,     # reused from another query type
    "slots_from_blend": 1,        # confidence-weighted selection
    "estimated_full_tokens": 300,
    "actual_tokens_used": 40,
    "savings_ratio": 0.87,
    "stitch": {
        "skeleton": str,
        "slot_fills": dict,
        "slot_sources": dict,
        "blend_candidates": dict,
        "gaps_detected": list,
        "slots_promoted": list,
        "answer_extracted": bool,
        "compound_formatted": bool,
        "multi_query": bool,
        "sub_results": list
    }
}
```

---

## Supporting Infrastructure

**CacheStore** — Single Redis connection, three key patterns:
- `template:{intent_id}` → serialized ResponseTemplate
- `slot:{slot_id}:{context_hash}` → serialized SlotRecord with
  time-decay
- `intent:{intent_id}` → serialized IntentCentroid
- `gap:{template_id}:{gap_type}` → gap event counter and aspect list
- `promoted:{template_id}` → set of already-promoted gap types

**Embedder** — In-memory SHA-256-keyed cache. Uses local
SentenceTransformer (all-MiniLM-L6-v2) or Gemini API. Identical strings
are never re-embedded within a session.

**LLM** — Single `llm_call(prompt, max_tokens)` function in
`utils/llm.py`. Routes to Ollama (local) or Gemini (remote). No module
ever calls the LLM API directly.

**GapLearner** — Classifies gaps via regex keyword patterns, stores
counts in Redis, promotes recurring gaps to template slots at threshold
3. Tracks promoted gap types per template to prevent duplicate
promotion.

**ClusterRouter** — Two-phase search with domain and subdomain rescue.
604 centroids across 24 k-means clusters. Online centroid averaging
updates the centroid embedding toward each matching query on every hit.

**Demo layer** — FastAPI with `POST /query` and `GET /stats`.
SavingsLog tracks hit rates, token savings, transfer counts, blend
counts, gap events, and slot promotions.

---

## Config flags reference

| Constant | Local default | Remote default | Purpose |
|---|---|---|---|
| `INTENT_SIMILARITY_THRESHOLD` | 0.55 | 0.90 | Cosine cutoff for intent match |
| `SLOT_CONFIDENCE_THRESHOLD` | 0.50 | 0.85 | Lower bound: uncertain below this |
| `SLOT_BLEND_THRESHOLD` | 0.65 | 0.92 | Upper bound: confident above this |
| `SLOT_TRANSFER_PENALTY` | 0.15 | 0.15 | Penalty for cross-template transfer |
| `SLOT_TRANSFER_ENABLED` | True | True | Enable cross-query slot transfer |
| `SLOT_BLEND_ENABLED` | True | True | Enable confidence-weighted blending |
| `GAP_PROMOTION_THRESHOLD` | 3 | 3 | Gap fires to trigger slot promotion |
| `GAP_LEARNING_ENABLED` | True | True | Enable gap pattern learning |
| `ANSWER_EXTRACTION_ENABLED` | True | True | Enable list answer extraction |
| `ANSWER_EXTRACTION_MIN_SCORE` | 3.0 | 3.0 | Minimum extraction confidence score |
| `UNCERTAIN_SLOT_FALLBACK_RATIO` | 0.5 | 0.5 | Uncertain fraction triggering fallback |
| `CONFIDENCE_DECAY_FACTOR` | 0.95 | 0.95 | Per-period decay multiplier |
| `CONFIDENCE_DECAY_DAYS` | 30 | 30 | Decay period length in days |

---

## Savings math

| Scenario | Tokens used |
|---|---|
| Full generation (cache miss) | ~300 |
| All slots confident (cache hit) | 0 |
| 2 of 6 slots uncertain | ~40 (two targeted LLM calls) |
| Slot transfer hits | ~20 (reused from other query types) |
| List answer extraction | 0 (item extracted from cached list) |

Typical mixed traffic: 55–75% reduction in output tokens.
Local mode savings are comparable — the lower thresholds mean more
cache hits but also more blend zone LLM calls, balancing out to a
similar net reduction.
