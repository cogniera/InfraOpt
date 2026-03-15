"""Gap pattern learning and slot promotion.

Tracks recurring gaps in cached responses and promotes frequently
occurring gap types to permanent template slots, so future queries
get them filled automatically without needing supplements.
"""

import logging
import re
from typing import List

from templatecache.config import GAP_LEARNING_ENABLED, GAP_PROMOTION_THRESHOLD
from templatecache.modules.cache_store import CacheStore

logger = logging.getLogger(__name__)

# ── Gap type classification patterns ─────────────────────────────────────

_TEMPORAL_RE = re.compile(
    r"\b(when|date|year|month|time|recent|latest|update|current|now|today|ago)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(compar\w*|vs|versus|differ\w*|better|worse|than|contrast|alternative)\b",
    re.IGNORECASE,
)
_QUANTITATIVE_RE = re.compile(
    r"\b(how (much|many|long|far|big|fast|often)|cost|price|size|number|count|rate|percent)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(why|caus\w*|reason|because|result|effect|impact|consequence|lead to)\b",
    re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"\b(how to|steps?|process|method|way to|procedure|guide)\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"\b(example|instance|sample|demonstrate|show me|such as|like what)\b",
    re.IGNORECASE,
)


class GapLearner:
    """Learns from detected gaps and promotes recurring ones to template slots.

    Attributes:
        _cache_store: CacheStore instance for persistence.
        _savings_log: Optional SavingsLog for recording promotion events.
    """

    def __init__(self, cache_store: CacheStore, savings_log=None):
        """Initialise with a cache store and optional savings log.

        Args:
            cache_store: CacheStore for reading/writing gap data and templates.
            savings_log: Optional SavingsLog for recording promotion events.
        """
        self._cache_store = cache_store
        self._savings_log = savings_log

    def classify_gap(self, aspect: str) -> str:
        """Classify a gap aspect into a semantic type.

        Classification order (first match wins):
        1. temporal — time-related questions
        2. comparison — comparing entities
        3. quantitative — amounts, sizes, costs
        4. causal — why/because questions
        5. procedural — how-to questions
        6. example — asking for examples
        7. elaboration — everything else

        Args:
            aspect: The query aspect text that triggered the gap.

        Returns:
            One of: 'temporal', 'comparison', 'quantitative', 'causal',
            'procedural', 'example', 'elaboration'.
        """
        if _TEMPORAL_RE.search(aspect):
            return "temporal"
        if _COMPARISON_RE.search(aspect):
            return "comparison"
        if _QUANTITATIVE_RE.search(aspect):
            return "quantitative"
        if _CAUSAL_RE.search(aspect):
            return "causal"
        if _PROCEDURAL_RE.search(aspect):
            return "procedural"
        if _EXAMPLE_RE.search(aspect):
            return "example"
        return "elaboration"

    def store_gap(self, template_id: str, gap_type: str, aspect: str) -> None:
        """Record a detected gap if gap learning is enabled.

        Args:
            template_id: The template intent ID.
            gap_type: Classified gap type from classify_gap().
            aspect: The query aspect text.

        Side effects:
            Writes to Redis via cache_store.store_gap().
        """
        if not GAP_LEARNING_ENABLED:
            return
        self._cache_store.store_gap(template_id, gap_type, aspect)

    def check_promotion(self, template_id: str) -> List[str]:
        """Check if any gap types should be promoted to template slots.

        For each gap type that has reached GAP_PROMOTION_THRESHOLD and
        has not already been promoted to this template: appends a new
        supplement slot to the template skeleton and slot order.

        Args:
            template_id: The template intent ID to check.

        Returns:
            List of newly promoted slot names (empty if none promoted).
        """
        if not GAP_LEARNING_ENABLED:
            return []

        counts = self._cache_store.get_gap_counts(template_id)
        if not counts:
            return []

        # Check what's already been promoted
        promoted_key = f"promoted:{template_id}"
        already_promoted = self._cache_store._redis.smembers(promoted_key)
        if isinstance(already_promoted, set) and already_promoted:
            already_promoted = {
                v.decode() if isinstance(v, bytes) else v for v in already_promoted
            }
        else:
            already_promoted = set()

        template = self._cache_store.get_template(template_id)
        if template is None:
            return []

        promoted_names: List[str] = []
        for gap_type, count in counts.items():
            if count < GAP_PROMOTION_THRESHOLD:
                continue
            if gap_type in already_promoted:
                continue

            # Determine supplement index
            existing = [s for s in template.slots if s.startswith(f"{gap_type}_supplement_")]
            idx = len(existing)
            slot_name = f"{gap_type}_supplement_{idx}"

            # Append to template
            template.skeleton += f"\n\n[{slot_name}]"
            template.slots.append(slot_name)
            template.dependency_graph[slot_name] = []

            # Mark as promoted
            self._cache_store._redis.sadd(promoted_key, gap_type)

            promoted_names.append(slot_name)
            logger.info(
                "Promoted gap type '%s' to slot '%s' on template '%s'",
                gap_type, slot_name, template_id,
            )

            # Log promotion event
            if self._savings_log is not None:
                self._savings_log.log_event({
                    "event_type": "slot_promoted",
                    "template_id": template_id,
                    "gap_type": gap_type,
                    "slot_name": slot_name,
                })

        # Save updated template if any promotions occurred
        if promoted_names:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._cache_store.write_back(template=template)
                )
            except RuntimeError:
                # No running event loop — write synchronously
                asyncio.run(self._cache_store.write_back(template=template))

        return promoted_names
