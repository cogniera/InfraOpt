"""Tests for IntentRouter module.

Covers: above-threshold query returns intent ID, below-threshold returns None,
variant selection responds to query keywords correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from templatecache.config import INTENT_SIMILARITY_THRESHOLD
from templatecache.models.intent import IntentCentroid
from templatecache.modules.cache_store import CacheStore
from templatecache.modules.router import IntentRouter


@pytest.fixture
def mock_cache_store():
    """Create a mock CacheStore."""
    return MagicMock(spec=CacheStore)


@pytest.fixture
def router(mock_cache_store):
    """Create an IntentRouter with mocked CacheStore."""
    return IntentRouter(mock_cache_store)


class TestRouteAboveThreshold:
    """Above-threshold query returns intent ID."""

    @patch("templatecache.modules.router.embed")
    @patch("templatecache.modules.router.cosine_similarity")
    def test_returns_intent_on_high_similarity(self, mock_cosine, mock_embed, router, mock_cache_store):
        """Query with high similarity returns the matching intent ID."""
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_cosine.return_value = 0.95  # Above threshold

        centroid = IntentCentroid(
            intent_id="greeting",
            centroid_embedding=[0.1, 0.2, 0.3],
            template_id="greeting",
            variant="short",
            query_count=5,
        )
        mock_cache_store.get_all_intent_centroids.return_value = [centroid]

        intent_id, variant = router.route("Hello there")
        assert intent_id == "greeting"


class TestRouteBelowThreshold:
    """Below-threshold query returns None."""

    @patch("templatecache.modules.router.embed")
    @patch("templatecache.modules.router.cosine_similarity")
    def test_returns_none_on_low_similarity(self, mock_cosine, mock_embed, router, mock_cache_store):
        """Query with low similarity returns None."""
        mock_embed.return_value = [0.9, 0.8, 0.7]
        mock_cosine.return_value = 0.5  # Below threshold

        centroid = IntentCentroid(
            intent_id="greeting",
            centroid_embedding=[0.1, 0.2, 0.3],
            template_id="greeting",
            variant="short",
            query_count=5,
        )
        mock_cache_store.get_all_intent_centroids.return_value = [centroid]

        intent_id, variant = router.route("Something completely different")
        assert intent_id is None
        assert variant is None

    @patch("templatecache.modules.router.embed")
    def test_returns_none_when_no_centroids(self, mock_embed, router, mock_cache_store):
        """Returns None when no centroids exist."""
        mock_embed.return_value = [0.1, 0.2]
        mock_cache_store.get_all_intent_centroids.return_value = []

        intent_id, variant = router.route("Hello")
        assert intent_id is None
        assert variant is None


class TestVariantSelection:
    """Variant selection responds to query keywords correctly."""

    @patch("templatecache.modules.router.embed")
    @patch("templatecache.modules.router.cosine_similarity")
    def test_detailed_variant_on_explain_keyword(self, mock_cosine, mock_embed, router, mock_cache_store):
        """Query with 'explain' keyword gets detailed variant."""
        mock_embed.return_value = [0.1, 0.2]
        mock_cosine.return_value = 0.95

        centroid = IntentCentroid(
            intent_id="explanation",
            centroid_embedding=[0.1, 0.2],
            template_id="explanation",
            variant="detailed",
            query_count=3,
        )
        mock_cache_store.get_all_intent_centroids.return_value = [centroid]

        intent_id, variant = router.route("Explain how this works")
        assert intent_id == "explanation"
        assert variant == "detailed"

    @patch("templatecache.modules.router.embed")
    @patch("templatecache.modules.router.cosine_similarity")
    def test_variant_disagreement_defaults_to_detailed(self, mock_cosine, mock_embed, router, mock_cache_store):
        """When query variant and centroid variant disagree, detailed wins."""
        mock_embed.return_value = [0.1, 0.2]
        mock_cosine.return_value = 0.95

        centroid = IntentCentroid(
            intent_id="listing",
            centroid_embedding=[0.1, 0.2],
            template_id="listing",
            variant="list",
            query_count=3,
        )
        mock_cache_store.get_all_intent_centroids.return_value = [centroid]

        # Short query (< 8 words, no keywords) → variant="short", centroid="list" → disagreement → "detailed"
        intent_id, variant = router.route("Hello")
        assert variant == "detailed"

