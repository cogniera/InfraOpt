"""Tests for GapLearner — gap classification, storage, and slot promotion."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from templatecache.modules.gap_learner import GapLearner
from templatecache.modules.cache_store import CacheStore
from templatecache.models.template import ResponseTemplate


@pytest.fixture
def mock_cache_store():
    """Create a mock CacheStore."""
    store = MagicMock(spec=CacheStore)
    store._redis = MagicMock()
    store._redis.smembers.return_value = set()
    return store


@pytest.fixture
def mock_savings_log():
    """Create a mock SavingsLog."""
    return MagicMock()


@pytest.fixture
def learner(mock_cache_store, mock_savings_log):
    """Create a GapLearner with mocked dependencies."""
    return GapLearner(mock_cache_store, savings_log=mock_savings_log)


class TestClassifyGap:
    """classify_gap returns correct gap types."""

    def test_temporal(self, learner):
        """Time-related aspect classified as temporal."""
        assert learner.classify_gap("when was this updated?") == "temporal"
        assert learner.classify_gap("what is the latest version?") == "temporal"
        assert learner.classify_gap("how recent is this data?") == "temporal"

    def test_comparison(self, learner):
        """Comparison aspect classified as comparison."""
        assert learner.classify_gap("how does it compare to X?") == "comparison"
        assert learner.classify_gap("what is the difference?") == "comparison"
        assert learner.classify_gap("is it better than Y?") == "comparison"

    def test_quantitative(self, learner):
        """Quantitative aspect classified as quantitative."""
        assert learner.classify_gap("how much does it cost?") == "quantitative"
        assert learner.classify_gap("what is the price?") == "quantitative"
        assert learner.classify_gap("how many users?") == "quantitative"

    def test_causal(self, learner):
        """Causal aspect classified as causal."""
        assert learner.classify_gap("why does this happen?") == "causal"
        assert learner.classify_gap("what caused it?") == "causal"
        assert learner.classify_gap("what is the reason?") == "causal"

    def test_procedural(self, learner):
        """Procedural aspect classified as procedural."""
        assert learner.classify_gap("how to install it?") == "procedural"
        assert learner.classify_gap("what are the steps?") == "procedural"

    def test_example(self, learner):
        """Example-requesting aspect classified as example."""
        assert learner.classify_gap("give me an example") == "example"
        assert learner.classify_gap("show me a sample") == "example"

    def test_elaboration_fallback(self, learner):
        """Unclassifiable aspect defaults to elaboration."""
        assert learner.classify_gap("tell me more about this") == "elaboration"
        assert learner.classify_gap("expand on that") == "elaboration"


class TestStoreGap:
    """store_gap delegates to cache_store when enabled."""

    def test_stores_when_enabled(self, learner, mock_cache_store):
        """Gap is stored when GAP_LEARNING_ENABLED is True."""
        with patch("templatecache.modules.gap_learner.GAP_LEARNING_ENABLED", True):
            learner.store_gap("t1", "temporal", "when?")
        mock_cache_store.store_gap.assert_called_once_with("t1", "temporal", "when?")

    def test_no_store_when_disabled(self, learner, mock_cache_store):
        """Gap is not stored when GAP_LEARNING_ENABLED is False."""
        with patch("templatecache.modules.gap_learner.GAP_LEARNING_ENABLED", False):
            learner.store_gap("t1", "temporal", "when?")
        mock_cache_store.store_gap.assert_not_called()


class TestCheckPromotion:
    """check_promotion promotes gap types that reach threshold."""

    def test_promotes_at_threshold(self, learner, mock_cache_store, mock_savings_log):
        """Gap type promoted when count reaches GAP_PROMOTION_THRESHOLD."""
        mock_cache_store.get_gap_counts.return_value = {"temporal": 3}
        template = ResponseTemplate(
            intent_id="t1",
            skeleton="Base response.",
            slots=[],
            dependency_graph={},
            variant="short",
        )
        mock_cache_store.get_template.return_value = template

        with patch("templatecache.modules.gap_learner.GAP_PROMOTION_THRESHOLD", 3):
            promoted = learner.check_promotion("t1")

        assert len(promoted) == 1
        assert promoted[0] == "temporal_supplement_0"
        assert "temporal_supplement_0" in template.slots
        assert "[temporal_supplement_0]" in template.skeleton
        mock_savings_log.log_event.assert_called_once()
        event = mock_savings_log.log_event.call_args[0][0]
        assert event["event_type"] == "slot_promoted"

    def test_no_promotion_below_threshold(self, learner, mock_cache_store):
        """No promotion when count is below threshold."""
        mock_cache_store.get_gap_counts.return_value = {"temporal": 1}
        with patch("templatecache.modules.gap_learner.GAP_PROMOTION_THRESHOLD", 3):
            promoted = learner.check_promotion("t1")
        assert promoted == []

    def test_no_double_promotion(self, learner, mock_cache_store):
        """Same gap type is not promoted twice to the same template."""
        mock_cache_store.get_gap_counts.return_value = {"temporal": 5}
        mock_cache_store._redis.smembers.return_value = {b"temporal"}
        template = ResponseTemplate(
            intent_id="t1",
            skeleton="Base.",
            slots=[],
            dependency_graph={},
            variant="short",
        )
        mock_cache_store.get_template.return_value = template

        with patch("templatecache.modules.gap_learner.GAP_PROMOTION_THRESHOLD", 3):
            promoted = learner.check_promotion("t1")

        assert promoted == []

    def test_disabled_returns_empty(self, learner):
        """Returns empty list when GAP_LEARNING_ENABLED is False."""
        with patch("templatecache.modules.gap_learner.GAP_LEARNING_ENABLED", False):
            promoted = learner.check_promotion("t1")
        assert promoted == []

    def test_multiple_gap_types_promoted(self, learner, mock_cache_store, mock_savings_log):
        """Multiple gap types can be promoted in one check."""
        mock_cache_store.get_gap_counts.return_value = {"temporal": 3, "comparison": 4}
        template = ResponseTemplate(
            intent_id="t1",
            skeleton="Base.",
            slots=[],
            dependency_graph={},
            variant="short",
        )
        mock_cache_store.get_template.return_value = template

        with patch("templatecache.modules.gap_learner.GAP_PROMOTION_THRESHOLD", 3):
            promoted = learner.check_promotion("t1")

        assert len(promoted) == 2
        assert "temporal_supplement_0" in promoted
        assert "comparison_supplement_0" in promoted

