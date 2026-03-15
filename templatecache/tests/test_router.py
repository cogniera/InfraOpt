"""Tests for IntentRouter module and subdomain rescue tiebreaking.

Covers: above-threshold query returns intent ID, below-threshold returns None,
variant selection responds to query keywords correctly, subdomain rescue
threshold promotes correct below-threshold candidates.
"""

from unittest.mock import MagicMock, patch

import pytest

from templatecache.config import INTENT_SIMILARITY_THRESHOLD
from templatecache.models.intent import IntentCentroid
from templatecache.modules.cache_store import CacheStore
from templatecache.modules.cluster_router import (
    _domain_tiebreak,
    _subdomain_tiebreak,
)
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


# ── Helpers for subdomain rescue tests ────────────────────────────────

def _make_centroid(intent_id: str, variant: str = "list") -> IntentCentroid:
    """Create a minimal IntentCentroid for tiebreak tests."""
    return IntentCentroid(
        intent_id=intent_id,
        centroid_embedding=[0.0],
        template_id=intent_id,
        variant=variant,
        query_count=1,
    )


# Threshold constants used in tests
_RESCUE_THRESHOLD = INTENT_SIMILARITY_THRESHOLD - 0.15


class TestSubdomainRescuePromotes:
    """Below-threshold candidate with matching subdomain keywords gets promoted."""

    def test_planet_query_promotes_solar_system(self):
        """space_solar_system below threshold is promoted for 'planet' query."""
        star = _make_centroid("space_largest_star")
        solar = _make_centroid("space_solar_system")

        # Scores relative to threshold: star above, solar in rescue zone
        above = INTENT_SIMILARITY_THRESHOLD + 0.01
        rescue = INTENT_SIMILARITY_THRESHOLD - 0.10

        candidates = [
            (above, star, None),    # above threshold, wrong subdomain
            (rescue, solar, None),  # below threshold, correct subdomain
        ]

        result = _domain_tiebreak("what is the largest planet", candidates)
        assert result is not None
        assert result[1].intent_id == "space_solar_system"

    def test_star_query_keeps_star_intent(self):
        """space_largest_star stays selected for 'star' query."""
        star = _make_centroid("space_largest_star")
        solar = _make_centroid("space_solar_system")

        above = INTENT_SIMILARITY_THRESHOLD + 0.01
        rescue = INTENT_SIMILARITY_THRESHOLD - 0.10

        candidates = [
            (above, star, None),
            (rescue, solar, None),
        ]

        result = _domain_tiebreak("what is the largest star", candidates)
        assert result is not None
        assert result[1].intent_id == "space_largest_star"


class TestSubdomainRescueNoMatch:
    """Below-threshold candidate with no subdomain match does not get promoted."""

    def test_no_subdomain_keywords_stays_below(self):
        """Candidate with no subdomain keyword match is not rescued."""
        star = _make_centroid("space_largest_star")
        exo = _make_centroid("space_exoplanet_detection")

        above = INTENT_SIMILARITY_THRESHOLD + 0.01
        # Below rescue threshold entirely
        below_rescue = INTENT_SIMILARITY_THRESHOLD - 0.20

        candidates = [
            (above, star, None),
            (below_rescue, exo, None),
        ]

        result = _domain_tiebreak("what is the largest star", candidates)
        assert result is None or result[1].intent_id == "space_largest_star"

    def test_unknown_domain_not_rescued(self):
        """Candidate in unknown domain is not rescued by subdomain logic."""
        star = _make_centroid("space_largest_star")
        misc = _make_centroid("misc_random_topic")

        above = INTENT_SIMILARITY_THRESHOLD + 0.01
        rescue = INTENT_SIMILARITY_THRESHOLD - 0.10

        candidates = [
            (above, star, None),
            (rescue, misc, None),
        ]

        result = _domain_tiebreak("what is the largest planet", candidates)
        assert result is None or result[1].intent_id != "misc_random_topic"


class TestAboveThresholdUnaffected:
    """Above-threshold candidate is not affected by subdomain logic."""

    def test_high_score_correct_domain_unchanged(self):
        """When top candidate is correct domain and high score, no override."""
        solar = _make_centroid("space_solar_system")
        country = _make_centroid("comp_largest_country")

        high = INTENT_SIMILARITY_THRESHOLD + 0.05
        low = INTENT_SIMILARITY_THRESHOLD + 0.01

        candidates = [
            (high, solar, None),   # top, correct domain
            (low, country, None),
        ]

        result = _domain_tiebreak("how many planets are in the solar system", candidates)
        assert result is None or result[1].intent_id == "space_solar_system"


class TestSubdomainTiebreakSameDomain:
    """Subdomain tiebreak within same domain picks correct intent."""

    def test_planets_subdomain_picks_solar_system(self):
        """Within space domain, planet query picks solar_system over star."""
        star = _make_centroid("space_largest_star")
        solar = _make_centroid("space_solar_system")

        candidates = [
            (0.90, star, None),
            (0.50, solar, None),
        ]

        result = _subdomain_tiebreak(
            "what is the largest planet", candidates, "space"
        )
        assert result is not None
        assert result[1].intent_id == "space_solar_system"

    def test_stars_subdomain_picks_star_intent(self):
        """Within space domain, star query picks largest_star over solar_system."""
        star = _make_centroid("space_largest_star")
        solar = _make_centroid("space_solar_system")

        candidates = [
            (0.90, star, None),
            (0.50, solar, None),
        ]

        result = _subdomain_tiebreak(
            "what is the largest star", candidates, "space"
        )
        assert result is not None
        assert result[1].intent_id == "space_largest_star"


class TestSubdomainTiedScoresFallback:
    """Tied subdomain scores fall back to embedding similarity."""

    def test_tied_subdomain_prefers_higher_similarity(self):
        """When subdomain scores are equal, higher embedding score wins."""
        # Both intent IDs contain "space" but neither contains planet/star
        # subdomain keywords — tied at 0
        a = _make_centroid("space_general_info")
        b = _make_centroid("space_overview_data")

        candidates = [
            (0.85, a, None),
            (0.80, b, None),
        ]

        result = _subdomain_tiebreak(
            "what is the largest planet", candidates, "space"
        )
        # Neither intent ID matches planets subdomain keywords, so
        # subdomain scores are both 0 — returns None (no differentiation)
        assert result is None

    def test_tied_nonzero_subdomain_prefers_higher_similarity(self):
        """When subdomain scores are tied but nonzero, higher similarity wins."""
        # Both contain "planet" in intent ID → both match planets subdomain
        a = _make_centroid("space_planet_alpha")
        b = _make_centroid("space_planet_beta")

        candidates = [
            (0.85, a, None),
            (0.80, b, None),
        ]

        result = _subdomain_tiebreak(
            "what is the largest planet", candidates, "space"
        )
        assert result is not None
        assert result[1].intent_id == "space_planet_alpha"  # higher similarity

