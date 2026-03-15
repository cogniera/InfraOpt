"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import logging

import httpx

from templatecache.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call via local Ollama. All modules must use this function exclusively.

    Args:
        prompt: The prompt to send to the LLM.
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

