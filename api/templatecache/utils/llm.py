"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import logging

from openai import OpenAI

from templatecache.config import OPENAI_API_KEY, OPENAI_LLM_MODEL

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on first use."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call. All modules must use this function exclusively."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

