"""Embedding utilities. All similarity comparisons must use cosine_similarity() from here."""

import hashlib
from typing import Dict, List

import httpx
import numpy as np

from templatecache.config import HF_API_TOKEN, HF_EMBEDDING_MODEL

_cache: Dict[str, List[float]] = {}

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_EMBEDDING_MODEL}"


def _embed_single(text: str) -> List[float]:
    """Embed a single string using HuggingFace Inference API.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(HF_INFERENCE_URL, json={"inputs": text}, headers=headers)
        resp.raise_for_status()
        return resp.json()


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
        # Batch call — HF Inference API supports list inputs
        headers = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                HF_INFERENCE_URL,
                json={"inputs": uncached_texts},
                headers=headers,
            )
            resp.raise_for_status()
            vectors = resp.json()

        for j, vector in enumerate(vectors):
            idx = uncached_indices[j]
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
