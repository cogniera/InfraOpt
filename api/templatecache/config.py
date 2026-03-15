"""All tunable constants for TemplateCache. Never hardcode these elsewhere."""

import os

# HuggingFace Inference Endpoint — primary LLM provider
USE_HF_LLM: bool = os.getenv("USE_HF_LLM", "true").lower() in ("true", "1", "yes")
HF_ENDPOINT_URL: str = os.getenv("HF_ENDPOINT_URL", "https://qyt7893blb71b5d3.us-east-2.aws.endpoints.huggingface.cloud/v1")
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")

# Intent matching
INTENT_SIMILARITY_THRESHOLD: float = 0.90

# Slot confidence
SLOT_CONFIDENCE_THRESHOLD: float = 0.85
UNCERTAIN_SLOT_FALLBACK_RATIO: float = 0.5

# Gap detection — how well must the cached response match the query?
GAP_COVERAGE_THRESHOLD: float = 0.45  # per-aspect coverage for multi-aspect queries
RESPONSE_RELEVANCE_THRESHOLD: float = 0.35  # whole-query relevance

# Confidence decay
CONFIDENCE_DECAY_FACTOR: float = 0.95
CONFIDENCE_DECAY_DAYS: int = 30

# Model names (Gemini — used for embeddings)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Variant token thresholds
VARIANT_SHORT_MAX_TOKENS: int = 80
VARIANT_DETAILED_MIN_TOKENS: int = 200

# Redis
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
