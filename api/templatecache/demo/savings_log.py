"""SavingsLog — in-memory request log for tracking cache savings."""

from typing import Dict, List


class SavingsLog:
    """In-memory log of request results for aggregate statistics.

    Tracks cache hits, token savings, slot usage, transfers, blends,
    and template evolution across all requests.
    """

    def __init__(self) -> None:
        """Initialize an empty savings log."""
        self._entries: List[Dict] = []
        self._events: List[Dict] = []
        self._evolved_templates: set = set()

    def record(self, result: Dict) -> None:
        """Record a query result.

        Args:
            result: The full pipeline response dict from TemplateCache.query().

        Side effects:
            Appends to the in-memory log. Tracks evolved templates.
        """
        self._entries.append(result)
        # Track template evolution from stitch metadata
        stitch = result.get("stitch", {})
        promoted = stitch.get("slots_promoted", [])
        if promoted and result.get("intent_id"):
            self._evolved_templates.add(result["intent_id"])

    def log_event(self, event: Dict) -> None:
        """Record a system event (e.g. slot promotion).

        Args:
            event: Event dict with at least 'event_type' key.

        Side effects:
            Appends to the events log. Tracks evolved template IDs.
        """
        self._events.append(event)
        if event.get("event_type") == "slot_promoted" and event.get("template_id"):
            self._evolved_templates.add(event["template_id"])

    def stats(self) -> Dict:
        """Compute aggregate statistics from the log.

        Returns:
            Dict with keys: total_requests, cache_hit_rate,
            average_savings_ratio, total_tokens_saved,
            slots_served_from_cache, slots_served_from_inference,
            slots_served_from_transfer, slots_served_from_blend,
            templates_evolved.
        """
        total = len(self._entries)
        if total == 0:
            return {
                "total_requests": 0,
                "cache_hit_rate": 0.0,
                "average_savings_ratio": 0.0,
                "total_tokens_saved": 0,
                "slots_served_from_cache": 0,
                "slots_served_from_inference": 0,
                "slots_served_from_transfer": 0,
                "slots_served_from_blend": 0,
                "templates_evolved": len(self._evolved_templates),
            }

        hits = sum(1 for e in self._entries if e.get("cache_hit", False))
        total_savings_ratio = sum(e.get("savings_ratio", 0.0) for e in self._entries)
        total_tokens_saved = sum(
            e.get("estimated_full_tokens", 0) - e.get("actual_tokens_used", 0)
            for e in self._entries
        )
        slots_cache = sum(e.get("slots_from_cache", 0) for e in self._entries)
        slots_inference = sum(e.get("slots_from_inference", 0) for e in self._entries)
        slots_transfer = sum(e.get("slots_from_transfer", 0) for e in self._entries)
        slots_blend = sum(e.get("slots_from_blend", 0) for e in self._entries)

        return {
            "total_requests": total,
            "cache_hit_rate": hits / total,
            "average_savings_ratio": total_savings_ratio / total,
            "total_tokens_saved": total_tokens_saved,
            "slots_served_from_cache": slots_cache,
            "slots_served_from_inference": slots_inference,
            "slots_served_from_transfer": slots_transfer,
            "slots_served_from_blend": slots_blend,
            "templates_evolved": len(self._evolved_templates),
        }

    def history(self) -> List[Dict]:
        """Return per-request token savings history for charting.

        Returns:
            List of dicts with tokens_saved per request, in order.
        """
        return [
            {
                "tokens_saved": e.get("estimated_full_tokens", 0) - e.get("actual_tokens_used", 0),
                "cache_hit": e.get("cache_hit", False),
            }
            for e in self._entries
        ]

