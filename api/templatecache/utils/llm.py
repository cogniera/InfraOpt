"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import json
import logging
from typing import Optional

import httpx

from templatecache.config import (
    HF_API_TOKEN,
    HF_ENDPOINT_URL,
    LLM_MODEL,
    USE_HF_LLM,
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


async def _huggingface_call(prompt: str, max_tokens: int) -> str:
    """Call HuggingFace inference endpoint for LLM generation.

    Args:
        prompt: The prompt to send.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The generated text from the HuggingFace endpoint.

    Side effects:
        Makes an HTTP call to the HuggingFace inference endpoint.
    """
    url = f"{HF_ENDPOINT_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


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
        Makes an API call to HuggingFace endpoint or Google Gemini.
    """
    if USE_HF_LLM:
        return await _huggingface_call(prompt, max_tokens)
    return await _gemini_call(prompt, max_tokens)
