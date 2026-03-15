"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import logging

import httpx

from templatecache.config import (
    HF_API_TOKEN,
    HF_ENDPOINT_URL,
)

logger = logging.getLogger(__name__)


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call via HuggingFace inference endpoint.

    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The LLM response text.
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
