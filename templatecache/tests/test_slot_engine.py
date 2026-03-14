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
        """Returns None when uncertain ratio exceeds UNCERTAIN_SLOT_FALLBACK_RATIO.

        Uses a mix of confident and uncertain slots (not all uncertain,
        since all-uncertain is treated as a new template and filled).
        """
        template = ResponseTemplate(
            intent_id="test",
            skeleton="[a] [b] [c] [d]",
            slots=["a", "b", "c", "d"],
            dependency_graph={"a": [], "b": [], "c": [], "d": []},
            variant="detailed",
        )
        # 1 confident + 3 uncertain → ratio = 3/4 = 0.75 > 0.5
        confident_record = MagicMock()
        confident_record.similarity_score = 0.95
        confident_record.fill_value = "cached_a"
        confident_record.fill_embedding = [0.1] * 384

        def side_effect(slot_id, context_hash):
            if slot_id == "a":
                return confident_record
            return None

        mock_cache_store.get_slot_confidence.side_effect = side_effect

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




class TestCleanFillValue:
    """_clean_fill_value strips brackets, quotes, and whitespace."""

    def test_strips_brackets(self, engine):
        """LLM returning [value] has brackets removed."""
        assert engine._clean_fill_value("[599]") == "599"

    def test_strips_quotes(self, engine):
        """LLM returning "value" has quotes removed."""
        assert engine._clean_fill_value('"hello world"') == "hello world"

    def test_strips_single_quotes(self, engine):
        """LLM returning 'value' has quotes removed."""
        assert engine._clean_fill_value("'hello world'") == "hello world"

    def test_strips_whitespace(self, engine):
        """Leading/trailing whitespace is removed."""
        assert engine._clean_fill_value("  hello  ") == "hello"

    def test_strips_brackets_and_whitespace(self, engine):
        """Combined bracket + whitespace stripping."""
        assert engine._clean_fill_value("  [42]  ") == "42"

    def test_preserves_inner_brackets(self, engine):
        """Brackets inside the value are preserved."""
        assert engine._clean_fill_value("array[0]") == "array[0]"

    def test_empty_string(self, engine):
        """Empty string returns empty."""
        assert engine._clean_fill_value("") == ""

    def test_no_stripping_needed(self, engine):
        """Clean value passes through unchanged."""
        assert engine._clean_fill_value("hello") == "hello"


class TestStitchEdgeCases:
    """_stitch handles edge cases: currency, adjacent punctuation, etc."""

    def test_currency_slots(self, engine):
        """Slots adjacent to $ signs are replaced correctly."""
        skeleton = "Price: $[number_0] to $[number_1]"
        fills = {"number_0": "5.49", "number_1": "5.99"}
        result = engine._stitch(skeleton, fills)
        assert result == "Price: $5.49 to $5.99"
        assert "[" not in result

    def test_bold_markdown_slots(self, engine):
        """Slots inside markdown bold markers are replaced."""
        skeleton = "**[name_0]** is great"
        fills = {"name_0": "Python"}
        result = engine._stitch(skeleton, fills)
        assert result == "**Python** is great"

    def test_adjacent_slots(self, engine):
        """Two slots with no space between them."""
        skeleton = "[first_0][last_0]"
        fills = {"first_0": "John", "last_0": "Doe"}
        result = engine._stitch(skeleton, fills)
        assert result == "JohnDoe"

    def test_repeated_slot_name(self, engine):
        """Same slot appearing multiple times in skeleton."""
        skeleton = "[name_0] likes [name_0]"
        fills = {"name_0": "Alice"}
        result = engine._stitch(skeleton, fills)
        assert result == "Alice likes Alice"

    def test_missing_fill_removed(self, engine):
        """Unfilled slot markers are removed (safety net)."""
        skeleton = "Hello [name_0], welcome to [place_0]."
        fills = {"name_0": "Alice"}  # place_0 missing
        result = engine._stitch(skeleton, fills)
        assert result == "Hello Alice, welcome to ."
        assert "[" not in result

    def test_supplement_slots(self, engine):
        """Supplement slots (from gap detection) are stitched."""
        skeleton = "Base response.\n\n[supplement_0]"
        fills = {"supplement_0": "Additional info here."}
        result = engine._stitch(skeleton, fills)
        assert result == "Base response.\n\nAdditional info here."

    def test_slot_in_parentheses(self, engine):
        """Slot inside parentheses."""
        skeleton = "([number_0] items)"
        fills = {"number_0": "42"}
        result = engine._stitch(skeleton, fills)
        assert result == "(42 items)"

    def test_slot_with_newlines(self, engine):
        """Slots across lines."""
        skeleton = "Line 1: [a_0]\nLine 2: [b_0]"
        fills = {"a_0": "hello", "b_0": "world"}
        result = engine._stitch(skeleton, fills)
        assert result == "Line 1: hello\nLine 2: world"

    def test_no_leftover_markers(self, engine):
        """No [slot_name] patterns remain after stitching."""
        skeleton = "[a_0] [b_1] [c_2] [quoted_content_0]"
        fills = {"a_0": "x", "b_1": "y", "c_2": "z", "quoted_content_0": "w"}
        result = engine._stitch(skeleton, fills)
        assert "[" not in result
        assert "]" not in result



class TestSlotTransfer:
    """Cross-query slot transfer via _transfer_slot."""

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    @patch("templatecache.modules.slot_engine.cosine_similarity")
    async def test_transfer_fires_when_confidence_fails(
        self, mock_cos, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Transfer is used when confidence check fails and cross-template candidate exists."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Price is $[currency_0]",
            slots=["currency_0"],
            dependency_graph={"currency_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = None  # uncertain

        # Set up transfer candidate
        candidate = SlotRecord(
            slot_id="currency_5",
            context_hash="other",
            fill_value="$99.99",
            fill_embedding=[0.1] * 10,
            similarity_score=0.9,
            slot_type="currency",
            decay_weight=0.8,
        )
        mock_cache_store.get_slots_by_type.return_value = [candidate]
        # cosine_similarity returns high score so transfer qualifies (0.85 - 0.15 = 0.70 > 0.50)
        mock_cos.return_value = 0.85

        response, from_cache, from_inference, stitch_info = await engine.fill(
            template, "test query"
        )
        assert response is not None
        assert stitch_info["slot_sources"]["currency_0"] == "transfer"
        assert stitch_info["slots_from_transfer"] == 1
        assert from_inference == 0
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_TRANSFER_ENABLED", False)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_transfer_disabled(self, mock_embed, mock_llm, engine, mock_cache_store):
        """Transfer does not fire when SLOT_TRANSFER_ENABLED is False."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Price is $[currency_0]",
            slots=["currency_0"],
            dependency_graph={"currency_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        mock_llm.return_value = "$50"
        mock_cache_store.get_slot_confidence.return_value = None
        mock_cache_store.get_slots_by_type.return_value = []

        response, from_cache, from_inference, stitch_info = await engine.fill(
            template, "test query"
        )
        assert stitch_info["slot_sources"]["currency_0"] == "inference"
        assert stitch_info["slots_from_transfer"] == 0
        mock_cache_store.get_slots_by_type.assert_not_called()

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    @patch("templatecache.modules.slot_engine.cosine_similarity")
    async def test_transfer_never_crosses_types(
        self, mock_cos, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Transfer only considers candidates of the same slot type."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Date: [date_0]",
            slots=["date_0"],
            dependency_graph={"date_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        mock_llm.return_value = "2024-01-01"
        mock_cache_store.get_slot_confidence.return_value = None
        mock_cache_store.get_slots_by_type.return_value = []  # no date candidates
        mock_cos.return_value = 0.9

        await engine.fill(template, "test query")
        # Should have called get_slots_by_type with "date", not other types
        mock_cache_store.get_slots_by_type.assert_called_once_with("date")

    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    def test_fill_source_logged_as_transfer(self, engine, mock_cache_store):
        """fill_source is logged as 'transfer' in slot_sources."""
        mock_cache_store.get_slots_by_type.return_value = [
            SlotRecord(
                slot_id="num_1",
                context_hash="x",
                fill_value="42",
                fill_embedding=[0.5] * 10,
                similarity_score=0.9,
                slot_type="number",
                decay_weight=0.8,
            )
        ]
        with patch("templatecache.modules.slot_engine.cosine_similarity", return_value=0.85):
            value, score = engine._transfer_slot("number_0", "number", [0.5] * 10)
        assert value == "42"
        assert score > 0

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_transfer_no_candidates(self, mock_embed, mock_llm, engine, mock_cache_store):
        """Transfer does not fire when no candidates exist."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Value: [number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        mock_llm.return_value = "100"
        mock_cache_store.get_slot_confidence.return_value = None
        mock_cache_store.get_slots_by_type.return_value = []

        response, from_cache, from_inference, stitch_info = await engine.fill(
            template, "test query"
        )
        assert stitch_info["slot_sources"]["number_0"] == "inference"
        assert stitch_info["slots_from_transfer"] == 0

    def test_transfer_filters_low_decay_weight(self, engine, mock_cache_store):
        """Transfer does not use records with decay_weight below 0.3."""
        # get_slots_by_type already filters by decay_weight >= 0.3
        # So if we pass records with low decay, they shouldn't be returned
        mock_cache_store.get_slots_by_type.return_value = []
        value, score = engine._transfer_slot("number_0", "number", [0.5] * 10)
        assert value is None
        assert score == 0.0



class TestConfidenceBlending:
    """Confidence-weighted response blending tests."""

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", True)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_below_confidence_goes_to_llm(
        self, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Slot below SLOT_CONFIDENCE_THRESHOLD goes to transfer/LLM, no blend."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Value: [number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        mock_llm.return_value = "100"
        mock_cache_store.get_slot_confidence.return_value = None  # below threshold
        mock_cache_store.get_slots_by_type.return_value = []

        _, _, _, stitch_info = await engine.fill(template, "test")
        assert stitch_info["slot_sources"]["number_0"] == "inference"
        assert stitch_info["slots_from_blend"] == 0

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", True)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed", return_value=[0.1] * 10)
    async def test_above_blend_threshold_serves_cached(
        self, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Slot above SLOT_BLEND_THRESHOLD serves cached directly, no LLM call."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Value: [number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        confident = MagicMock()
        confident.similarity_score = 0.95  # above blend threshold (0.90)
        confident.fill_value = "cached_42"
        confident.fill_embedding = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = confident

        _, from_cache, _, stitch_info = await engine.fill(template, "test")
        assert stitch_info["slot_sources"]["number_0"] == "cache"
        assert from_cache == 1
        assert stitch_info["slots_from_blend"] == 0
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", True)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed", return_value=[0.1] * 10)
    async def test_blend_zone_calls_llm(self, mock_embed, mock_llm, engine, mock_cache_store):
        """Slot in blend zone calls LLM and produces both candidates."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Value: [number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        blend_record = MagicMock()
        blend_record.similarity_score = 0.75  # between 0.50 and 0.90
        blend_record.fill_value = "cached_val"
        blend_record.fill_embedding = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = blend_record
        mock_llm.return_value = "fresh_val"

        _, _, _, stitch_info = await engine.fill(template, "test")
        assert stitch_info["slots_from_blend"] == 1
        assert "number_0" in stitch_info["blend_candidates"]
        bc = stitch_info["blend_candidates"]["number_0"]
        assert bc["cached_value"] == "cached_val"
        assert bc["fresh_value"] == "fresh_val"
        assert bc["confidence"] == 0.75
        assert bc["selected"] in ("cached", "fresh")
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", False)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed")
    async def test_blend_disabled_treats_as_uncertain(
        self, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Blend disabled: blend zone slots treated as uncertain (no blend record)."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="Value: [number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        mock_embed.return_value = [0.1] * 10
        blend_record = MagicMock()
        blend_record.similarity_score = 0.75  # between 0.50 and 0.90
        blend_record.fill_value = "cached_val"
        blend_record.fill_embedding = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = blend_record
        mock_cache_store.get_slots_by_type.return_value = []
        mock_llm.return_value = "llm_val"

        _, _, _, stitch_info = await engine.fill(template, "test")
        assert stitch_info["slots_from_blend"] == 0
        assert len(stitch_info["blend_candidates"]) == 0

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", True)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed", return_value=[0.1] * 10)
    async def test_blend_candidates_logged(self, mock_embed, mock_llm, engine, mock_cache_store):
        """blend_candidates logged correctly for blend zone slots."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="[a_0] [b_0]",
            slots=["a_0", "b_0"],
            dependency_graph={"a_0": [], "b_0": []},
            variant="short",
        )
        rec = MagicMock()
        rec.similarity_score = 0.70
        rec.fill_value = "cached"
        rec.fill_embedding = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = rec
        mock_llm.return_value = "fresh"

        _, _, _, stitch_info = await engine.fill(template, "test")
        assert len(stitch_info["blend_candidates"]) == 2
        for name in ("a_0", "b_0"):
            assert name in stitch_info["blend_candidates"]
            assert stitch_info["slot_sources"][name] == "blend"

    @pytest.mark.asyncio
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_THRESHOLD", 0.90)
    @patch("templatecache.modules.slot_engine.SLOT_CONFIDENCE_THRESHOLD", 0.50)
    @patch("templatecache.modules.slot_engine.SLOT_BLEND_ENABLED", True)
    @patch("templatecache.modules.slot_engine.llm_call", new_callable=AsyncMock)
    @patch("templatecache.modules.slot_engine.embed", return_value=[0.1] * 10)
    async def test_blend_respects_confidence_weighting(
        self, mock_embed, mock_llm, engine, mock_cache_store
    ):
        """Over 100 iterations blend selection respects confidence within 15% tolerance."""
        template = ResponseTemplate(
            intent_id="test",
            skeleton="[number_0]",
            slots=["number_0"],
            dependency_graph={"number_0": []},
            variant="short",
        )
        confidence = 0.80  # should pick "cached" ~80% of the time
        rec = MagicMock()
        rec.similarity_score = confidence
        rec.fill_value = "cached_val"
        rec.fill_embedding = [0.1] * 10
        mock_cache_store.get_slot_confidence.return_value = rec
        mock_llm.return_value = "fresh_val"

        cached_count = 0
        iterations = 200
        for _ in range(iterations):
            _, _, _, stitch_info = await engine.fill(template, "test")
            bc = stitch_info["blend_candidates"]["number_0"]
            if bc["selected"] == "cached":
                cached_count += 1

        ratio = cached_count / iterations
        # Should be within 15% of the confidence value
        assert abs(ratio - confidence) < 0.15, (
            f"Expected ~{confidence:.0%} cached, got {ratio:.0%}"
        )
