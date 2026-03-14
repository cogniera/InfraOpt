"""Tests for SlotEngine module.

Covers: confident slots skip LLM call, uncertain slots invoke LLM,
fallback triggers when uncertain ratio exceeds threshold,
_stitch() replaces all markers with no leftovers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from templatecache.config import SLOT_CONFIDENCE_THRESHOLD, UNCERTAIN_SLOT_FALLBACK_RATIO
from templatecache.models.slot import SlotRecord
from templatecache.models.template import ResponseTemplate
from templatecache.modules.cache_store import CacheStore
from templatecache.modules.slot_engine import SlotEngine


@pytest.fixture
def mock_cache_store():
    """Create a mock CacheStore."""
    mock = MagicMock(spec=CacheStore)
    mock.write_back = AsyncMock()
    return mock


@pytest.fixture
def engine(mock_cache_store):
    """Create a SlotEngine with mocked CacheStore."""
    return SlotEngine(mock_cache_store)


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return ResponseTemplate(
        intent_id="test",
        skeleton="Hello [name], welcome to [place].",
        slots=["name", "place"],
        dependency_graph={"name": [], "place": []},
        variant="short",
    )


class TestStitch:
    """_stitch() replaces all markers with no leftovers."""

    def test_replaces_all_markers(self, engine):
        """All [slot_name] markers are replaced with fill values."""
        skeleton = "Hello [name], welcome to [place]."
        fills = {"name": "Alice", "place": "Wonderland"}
        result = engine._stitch(skeleton, fills)
        assert result == "Hello Alice, welcome to Wonderland."
        assert "[" not in result
        assert "]" not in result

    def test_empty_fills(self, engine):
        """Skeleton with no markers returns unchanged."""
        skeleton = "Hello world."
        result = engine._stitch(skeleton, {})
        assert result == "Hello world."


class TestConfidentSlotsSkipLLM:
    """Confident slots skip LLM call."""

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_all_confident_skips_llm(self, mock_embed, mock_llm, engine, mock_cache_store, sample_template):
        """When all slots are confident, no LLM calls are made."""
        mock_embed.return_value = [0.1, 0.2]

        def slot_confidence(slot_id, ctx_hash):
            return SlotRecord(
                slot_id=slot_id,
                context_hash=ctx_hash,
                fill_value="cached_value",
                fill_embedding=[0.1],
                similarity_score=0.95,  # Above threshold
            )

        mock_cache_store.get_slot_confidence.side_effect = slot_confidence

        response, from_cache, from_inference, stitch_info = await engine.fill(sample_template, "test query")
        assert response is not None
        assert from_cache == 2
        assert from_inference == 0
        mock_llm.assert_not_called()


class TestUncertainSlotsInvokeLLM:
    """Uncertain slots invoke LLM."""

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_uncertain_slot_calls_llm(self, mock_embed, mock_llm, engine, mock_cache_store, sample_template):
        """When one slot is uncertain, LLM is called for it."""
        mock_embed.return_value = [0.1, 0.2]
        mock_llm.return_value = "generated_value"

        def slot_confidence(slot_id, ctx_hash):
            if slot_id == "name":
                return SlotRecord(
                    slot_id=slot_id,
                    context_hash=ctx_hash,
                    fill_value="cached_name",
                    similarity_score=0.95,
                )
            return None  # place is uncertain

        mock_cache_store.get_slot_confidence.side_effect = slot_confidence

        response, from_cache, from_inference, stitch_info = await engine.fill(sample_template, "test query")
        assert response is not None
        assert from_cache == 1
        assert from_inference == 1
        mock_llm.assert_called_once()


class TestFallbackOnHighUncertainRatio:
    """Fallback triggers when uncertain ratio exceeds threshold."""

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    async def test_fallback_when_too_many_uncertain(self, mock_llm, engine, mock_cache_store):
        """Returns None when uncertain ratio exceeds UNCERTAIN_SLOT_FALLBACK_RATIO."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="[a] [b] [c] [d]",
            slots=["a", "b", "c", "d"],
            dependency_graph={"a": [], "b": [], "c": [], "d": []},
            variant="detailed",
        )
        # All slots return None (uncertain) → ratio = 4/4 = 1.0 > 0.5
        mock_cache_store.get_slot_confidence.return_value = None

        response, from_cache, from_inference, stitch_info = await engine.fill(template, "test query")
        assert response is None
        assert from_cache == 0
        assert from_inference == 0
        mock_llm.assert_not_called()


class TestDependencyOrder:
    """Slots are filled in dependency order."""

    def test_dependency_order_simple(self, engine):
        """Slots with dependencies come after their dependencies."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="[a] [b] [c]",
            slots=["a", "b", "c"],
            dependency_graph={"a": [], "b": ["a"], "c": ["b"]},
            variant="short",
        )
        order = engine._dependency_order(template)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

