"""Central LLM call interface. All LLM calls must go through llm_call() only."""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    """Return the shared OpenAI-compatible client, creating it on first use."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "123"),
            base_url="https://qyt7893blb71b5d3.us-east-2.aws.endpoints.huggingface.cloud/v1",
        )
    return _client


async def llm_call(prompt: str, max_tokens: int) -> str:
    """Make an LLM call. All modules must use this function exclusively.

    Args:
        prompt: The prompt to send to the LLM.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The LLM response text.

    Side effects:
        Makes an API call to the Hugging Face inference endpoint.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

