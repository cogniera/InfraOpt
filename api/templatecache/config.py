"""All tunable constants for TemplateCache. Never hardcode these elsewhere."""

import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

# Intent matching
INTENT_SIMILARITY_THRESHOLD: float = 0.90

# Slot confidence
SLOT_CONFIDENCE_THRESHOLD: float = 0.85

# Per-type confidence thresholds — overrides SLOT_CONFIDENCE_THRESHOLD per slot type.
# SLOT_CONFIDENCE_THRESHOLD remains the fallback for any type not in this dict.
SLOT_CONFIDENCE_THRESHOLDS: dict = {
    "currency": 0.85,
    "date": 0.82,
    "duration": 0.80,
    "numeric": 0.78,
    "named_entity": 0.80,
    "boilerplate": 0.50,
    "quoted_content": 0.65,
}

UNCERTAIN_SLOT_FALLBACK_RATIO: float = 0.5

# Gap detection — how well must the cached response match the query?
GAP_COVERAGE_THRESHOLD: float = 0.45  # per-aspect coverage for multi-aspect queries
RESPONSE_RELEVANCE_THRESHOLD: float = 0.35  # whole-query relevance

# Confidence decay
CONFIDENCE_DECAY_FACTOR: float = 0.95
CONFIDENCE_DECAY_DAYS: int = 30

# Cross-query slot transfer
SLOT_TRANSFER_ENABLED: bool = True
SLOT_TRANSFER_PENALTY: float = 0.15  # subtracted from similarity when transferring across templates

# Confidence-weighted response blending
SLOT_BLEND_ENABLED: bool = True
SLOT_BLEND_THRESHOLD: float = 0.92  # above this: serve cached directly

# Gap pattern learning
GAP_LEARNING_ENABLED: bool = True
GAP_PROMOTION_THRESHOLD: int = 3  # gap type occurrences before promoting to slot

# Answer extraction from list-style responses
ANSWER_EXTRACTION_ENABLED: bool = True
ANSWER_EXTRACTION_MIN_SCORE: float = 3.0

# OpenAI Embeddings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Variant token thresholds
VARIANT_SHORT_MAX_TOKENS: int = 80
VARIANT_DETAILED_MIN_TOKENS: int = 200

# Redis
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
