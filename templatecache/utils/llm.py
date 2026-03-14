"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import json
import logging
from typing import Optional

import httpx

from templatecache.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    USE_LOCAL_LLM,
)

logger = logging.getLogger(__name__)

_gemini_client = None


def _get_gemini_client():
    """Return the shared Gemini client, creating it on first use."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client()
    return _gemini_client


async def _ollama_call(prompt: str, max_tokens: int) -> str:
    """Call local Ollama instance for LLM generation.

    Args:
        prompt: The prompt to send.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The generated text from Ollama.

    Side effects:
        Makes an HTTP call to the local Ollama server.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


async def _gemini_call(prompt: str, max_tokens: int) -> str:
    """Call Google Gemini API for LLM generation.

    Args:
        prompt: The prompt to send.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The generated text from Gemini.

    Side effects:
        Makes an API call to Google Gemini.
    """
    from google.genai import types

    response = _get_gemini_client().models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call. All modules must use this function exclusively.

    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The LLM response text.

    Side effects:
        Makes an API call to Ollama (local) or Google Gemini (remote).
    """
    if USE_LOCAL_LLM:
        return await _ollama_call(prompt, max_tokens)
    return await _gemini_call(prompt, max_tokens)

