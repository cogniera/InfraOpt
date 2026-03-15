"""Embedding utilities. All similarity comparisons must use cosine_similarity() from here."""

import hashlib
from typing import Dict, List

import httpx
import numpy as np

from templatecache.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_cache: Dict[str, List[float]] = {}

OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"


def _embed_single(text: str) -> List[float]:
    """Embed a single string using OpenAI Embeddings API.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "input": text,
        "model": OPENAI_EMBEDDING_MODEL,
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(OPENAI_EMBEDDING_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def _cache_key(text: str) -> str:
    """Return a stable hash key for caching embeddings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(text: str) -> List[float]:
    """Embed a single string, returning from in-memory cache if available.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    key = _cache_key(text)
    if key in _cache:
        return _cache[key]
    vector = _embed_single(text)
    _cache[key] = vector
    return vector


def batch_embed(texts: List[str]) -> List[List[float]]:
    """Embed multiple strings, using cache where possible.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors in the same order as input.
    """
    results: List[List[float]] = [[] for _ in texts]
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    for i, text in enumerate(texts):
        key = _cache_key(text)
        if key in _cache:
            results[i] = _cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        payload = {
            "input": uncached_texts,
            "model": OPENAI_EMBEDDING_MODEL,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(OPENAI_EMBEDDING_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()["data"]

        for j, item in enumerate(data):
            idx = uncached_indices[j]
            vector = item["embedding"]
            _cache[_cache_key(uncached_texts[j])] = vector
            results[idx] = vector

    return results


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
