"""CacheStore — Redis-backed persistence for templates, slot records, and intent centroids.

All other modules talk to Redis exclusively through this class.
"""

import json
import math
from datetime import UTC, datetime
from typing import List, Optional

import redis

from templatecache.config import (
    CONFIDENCE_DECAY_DAYS,
    CONFIDENCE_DECAY_FACTOR,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    SLOT_CONFIDENCE_THRESHOLD,
    SLOT_CONFIDENCE_THRESHOLDS,
)
from templatecache.models.intent import IntentCentroid
from templatecache.models.slot import SlotRecord
from templatecache.models.template import ResponseTemplate


class CacheStore:
    """Redis-backed persistence layer.

    Three operations: get_template, get_slot_confidence, write_back.
    Redis connection is instantiated once in __init__ and reused.
    """

    def __init__(self) -> None:
        """Initialize CacheStore with a single Redis connection."""
        self._redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )

    def get_template(self, intent_id: str) -> Optional[ResponseTemplate]:
        """Retrieve a ResponseTemplate from Redis by intent ID.

        Args:
            intent_id: The intent identifier.

        Returns:
            ResponseTemplate if found, None otherwise. Missing keys return None
            without raising.
        """
        raw = self._redis.get(f"template:{intent_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        return ResponseTemplate(**data)

    def get_slot_confidence(self, slot_id: str, context_hash: str) -> Optional[SlotRecord]:
        """Retrieve a SlotRecord and apply time-based confidence decay.

        The effective confidence threshold is determined by the slot's type
        via ``SLOT_CONFIDENCE_THRESHOLDS.get(slot_type, SLOT_CONFIDENCE_THRESHOLD)``.
        The resolved threshold is stored on the returned record as
        ``effective_threshold`` for callers to use.

        Args:
            slot_id: The slot name.
            context_hash: Hash of the query context.

        Returns:
            SlotRecord with decayed similarity_score and effective_threshold
            if found, None otherwise. Missing keys return None without raising.
        """
        raw = self._redis.get(f"slot:{slot_id}:{context_hash}")
        if raw is None:
            return None
        data = json.loads(raw)
        record = SlotRecord(**data)

        # Apply time-based decay
        created = datetime.fromisoformat(record.created_at)
        # Ensure created is timezone-aware for comparison
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - created).days
        periods = age_days / CONFIDENCE_DECAY_DAYS
        decay = math.pow(CONFIDENCE_DECAY_FACTOR, periods)
        record.decay_weight = decay
        record.similarity_score = record.similarity_score * decay

        # Resolve type-specific confidence threshold
        slot_type = getattr(record, "slot_type", None)
        if slot_type is not None:
            record.effective_threshold = SLOT_CONFIDENCE_THRESHOLDS.get(
                slot_type, SLOT_CONFIDENCE_THRESHOLD
            )
        else:
            record.effective_threshold = SLOT_CONFIDENCE_THRESHOLD

        return record

    def get_intent_centroid(self, intent_id: str) -> Optional[IntentCentroid]:
        """Retrieve an IntentCentroid from Redis.

        Args:
            intent_id: The intent identifier.

        Returns:
            IntentCentroid if found, None otherwise. Missing keys return None
            without raising.
        """
        raw = self._redis.get(f"intent:{intent_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        return IntentCentroid(**data)

    def get_all_intent_centroids(self) -> List[IntentCentroid]:
        """Retrieve all IntentCentroids from Redis.

        Returns:
            List of all IntentCentroid objects stored in Redis.

        Side effects:
            Performs a SCAN over Redis keys matching 'intent:*'.
        """
        centroids: List[IntentCentroid] = []
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match="intent:*", count=100)
            for key in keys:
                raw = self._redis.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    centroids.append(IntentCentroid(**data))
            if cursor == 0:
                break
        return centroids

    async def write_back(
        self,
        template: Optional[ResponseTemplate] = None,
        slot_record: Optional[SlotRecord] = None,
        centroid: Optional[IntentCentroid] = None,
    ) -> None:
        """Persist template, slot record, or centroid to Redis.

        Write-back must never block the response path — callers should use
        asyncio.create_task(cache_store.write_back(...)).

        Args:
            template: Optional ResponseTemplate to persist.
            slot_record: Optional SlotRecord to persist.
            centroid: Optional IntentCentroid to persist.

        Side effects:
            Writes to Redis.
        """
        if template is not None:
            key = f"template:{template.intent_id}"
            self._redis.set(key, json.dumps(template.__dict__))
        if slot_record is not None:
            key = f"slot:{slot_record.slot_id}:{slot_record.context_hash}"
            self._redis.set(key, json.dumps(slot_record.__dict__))
        if centroid is not None:
            key = f"intent:{centroid.intent_id}"
            self._redis.set(key, json.dumps(centroid.__dict__))

    def get_slots_by_type(self, slot_type: str) -> List[SlotRecord]:
        """Retrieve all SlotRecords matching a given slot type.

        Scans all Redis keys matching ``slot:*``, deserializes each
        SlotRecord, and returns those whose ``slot_type`` matches the
        requested type. Records with ``decay_weight`` below 0.3 are
        filtered out.

        Args:
            slot_type: The slot type to filter by (e.g. 'currency', 'date').

        Returns:
            List of matching SlotRecord objects with decay_weight >= 0.3.
        """
        results: List[SlotRecord] = []
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match="slot:*", count=100)
            for key in keys:
                raw = self._redis.get(key)
                if raw is None:
                    continue
                data = json.loads(raw)
                record = SlotRecord(**data)
                if record.slot_type != slot_type:
                    continue
                # Apply time-based decay for filtering
                created = datetime.fromisoformat(record.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                age_days = (datetime.now(UTC) - created).days
                periods = age_days / CONFIDENCE_DECAY_DAYS
                decay = math.pow(CONFIDENCE_DECAY_FACTOR, periods)
                record.decay_weight = decay
                if record.decay_weight < 0.3:
                    continue
                results.append(record)
            if cursor == 0:
                break
        return results

    def store_gap(self, template_id: str, gap_type: str, aspect: str) -> None:
        """Record a detected gap for a template.

        Increments a counter at ``gap:{template_id}:{gap_type}`` and
        appends the aspect text to the stored list.

        Args:
            template_id: The template intent ID.
            gap_type: Classified gap type (e.g. 'temporal', 'comparison').
            aspect: The query aspect text that triggered the gap.

        Side effects:
            Writes to Redis key ``gap:{template_id}:{gap_type}``.
        """
        key = f"gap:{template_id}:{gap_type}"
        raw = self._redis.get(key)
        if raw is None:
            data = {"count": 0, "aspects": []}
        else:
            data = json.loads(raw)
        data["count"] = data.get("count", 0) + 1
        aspects = data.get("aspects", [])
        aspects.append(aspect)
        data["aspects"] = aspects
        self._redis.set(key, json.dumps(data))

    def get_gap_counts(self, template_id: str) -> dict:
        """Return all gap type counts for a template.

        Args:
            template_id: The template intent ID.

        Returns:
            Dict mapping gap_type to count (e.g. {'temporal': 3, 'comparison': 1}).
        """
        result: dict = {}
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(
                cursor=cursor, match=f"gap:{template_id}:*", count=100
            )
            for key in keys:
                raw = self._redis.get(key)
                if raw is None:
                    continue
                data = json.loads(raw)
                # key format: gap:{template_id}:{gap_type}
                parts = key.split(":")
                if len(parts) >= 3:
                    gap_type = parts[2]
                    result[gap_type] = data.get("count", 0)
            if cursor == 0:
                break
        return result


