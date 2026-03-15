"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import logging

import httpx

from templatecache.config import OPENAI_API_KEY, OPENAI_LLM_MODEL

logger = logging.getLogger(__name__)


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call via OpenAI API.

    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The LLM response text.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "model": OPENAI_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
