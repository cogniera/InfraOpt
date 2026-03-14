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

        Args:
            slot_id: The slot name.
            context_hash: Hash of the query context.

        Returns:
            SlotRecord with decayed similarity_score if found, None otherwise.
            Missing keys return None without raising.
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

