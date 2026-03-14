"""Tests for CacheStore module.

Covers: write and read round-trip for templates and slot records,
decay reduces score for records older than CONFIDENCE_DECAY_DAYS,
missing keys return None without raising.
"""

import json
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from templatecache.config import CONFIDENCE_DECAY_DAYS, CONFIDENCE_DECAY_FACTOR
from templatecache.models.intent import IntentCentroid
from templatecache.models.slot import SlotRecord
from templatecache.models.template import ResponseTemplate
from templatecache.modules.cache_store import CacheStore


@pytest.fixture
def mock_redis():
    """Create a mock Redis client with dict-based storage."""
    store = {}
    mock = MagicMock()
    mock.get = lambda key: store.get(key)
    mock.set = lambda key, value: store.__setitem__(key, value)

    def mock_scan(cursor=0, match="*", count=100):
        import fnmatch
        matched = [k for k in store if fnmatch.fnmatch(k, match)]
        return (0, matched)

    mock.scan = mock_scan
    return mock, store


@pytest.fixture
def cache_store(mock_redis):
    """Create a CacheStore with mocked Redis."""
    mock, _ = mock_redis
    cs = CacheStore.__new__(CacheStore)
    cs._redis = mock
    return cs


class TestTemplateRoundTrip:
    """Write and read round-trip for templates."""

    @pytest.mark.asyncio
    async def test_write_and_read_template(self, cache_store):
        """Template can be written and read back correctly."""
        template = ResponseTemplate(
            intent_id="test_intent",
            skeleton="Hello [name], welcome to [place].",
            slots=["name", "place"],
            dependency_graph={"name": [], "place": []},
            variant="short",
        )
        await cache_store.write_back(template=template)
        result = cache_store.get_template("test_intent")

        assert result is not None
        assert result.intent_id == "test_intent"
        assert result.skeleton == "Hello [name], welcome to [place]."
        assert result.slots == ["name", "place"]
        assert result.variant == "short"

    def test_missing_template_returns_none(self, cache_store):
        """Missing template key returns None without raising."""
        result = cache_store.get_template("nonexistent")
        assert result is None


class TestSlotRecordRoundTrip:
    """Write and read round-trip for slot records."""

    @pytest.mark.asyncio
    async def test_write_and_read_slot_record(self, cache_store):
        """Slot record can be written and read back correctly."""
        record = SlotRecord(
            slot_id="name",
            context_hash="abc123",
            fill_value="Alice",
            fill_embedding=[0.1, 0.2, 0.3],
            similarity_score=0.95,
        )
        await cache_store.write_back(slot_record=record)
        result = cache_store.get_slot_confidence("name", "abc123")

        assert result is not None
        assert result.slot_id == "name"
        assert result.fill_value == "Alice"

    def test_missing_slot_returns_none(self, cache_store):
        """Missing slot key returns None without raising."""
        result = cache_store.get_slot_confidence("missing", "hash")
        assert result is None


class TestConfidenceDecay:
    """Decay reduces score for records older than CONFIDENCE_DECAY_DAYS."""

    @pytest.mark.asyncio
    async def test_decay_applied_to_old_records(self, cache_store):
        """Records older than CONFIDENCE_DECAY_DAYS have reduced scores."""
        old_date = (datetime.now(UTC) - timedelta(days=CONFIDENCE_DECAY_DAYS * 2)).isoformat()
        record = SlotRecord(
            slot_id="price",
            context_hash="ctx1",
            fill_value="$100",
            fill_embedding=[0.5],
            similarity_score=1.0,
            created_at=old_date,
        )
        await cache_store.write_back(slot_record=record)
        result = cache_store.get_slot_confidence("price", "ctx1")

        assert result is not None
        # 2 periods of decay applied
        expected_score = 1.0 * math.pow(CONFIDENCE_DECAY_FACTOR, 2)
        assert abs(result.similarity_score - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_no_decay_for_fresh_records(self, cache_store):
        """Recently created records have minimal decay."""
        record = SlotRecord(
            slot_id="name",
            context_hash="ctx2",
            fill_value="Bob",
            fill_embedding=[0.1],
            similarity_score=0.9,
        )
        await cache_store.write_back(slot_record=record)
        result = cache_store.get_slot_confidence("name", "ctx2")

        assert result is not None
        # Score should be very close to original (0 days of decay)
        assert abs(result.similarity_score - 0.9) < 0.01


class TestIntentCentroidRoundTrip:
    """Write and read round-trip for intent centroids."""

    @pytest.mark.asyncio
    async def test_write_and_read_centroid(self, cache_store):
        """Centroid can be written and read back correctly."""
        centroid = IntentCentroid(
            intent_id="greeting",
            centroid_embedding=[0.1, 0.2, 0.3],
            template_id="greeting",
            variant="short",
            query_count=5,
        )
        await cache_store.write_back(centroid=centroid)
        result = cache_store.get_intent_centroid("greeting")

        assert result is not None
        assert result.intent_id == "greeting"
        assert result.variant == "short"
        assert result.query_count == 5

    def test_missing_centroid_returns_none(self, cache_store):
        """Missing centroid key returns None without raising."""
        result = cache_store.get_intent_centroid("nonexistent")
        assert result is None

