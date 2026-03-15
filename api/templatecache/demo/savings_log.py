"""SavingsLog — in-memory request log for tracking cache savings."""

from typing import Dict, List


class SavingsLog:
    """In-memory log of request results for aggregate statistics.

    Tracks cache hits, token savings, and slot usage across all requests.
    """

    def __init__(self) -> None:
        """Initialize an empty savings log."""
        self._entries: List[Dict] = []

    def record(self, result: Dict) -> None:
        """Record a query result.

        Args:
            result: The full pipeline response dict from TemplateCache.query().

        Side effects:
            Appends to the in-memory log.
        """
        self._entries.append(result)

    def stats(self) -> Dict:
        """Compute aggregate statistics from the log.

        Returns:
            Dict with keys: total_requests, cache_hit_rate,
            average_savings_ratio, total_tokens_saved,
            slots_served_from_cache, slots_served_from_inference.
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
            }

        hits = sum(1 for e in self._entries if e.get("cache_hit", False))
        total_savings_ratio = sum(e.get("savings_ratio", 0.0) for e in self._entries)
        total_tokens_saved = sum(
            e.get("estimated_full_tokens", 0) - e.get("actual_tokens_used", 0)
            for e in self._entries
        )
        slots_cache = sum(e.get("slots_from_cache", 0) for e in self._entries)
        slots_inference = sum(e.get("slots_from_inference", 0) for e in self._entries)

        return {
            "total_requests": total,
            "cache_hit_rate": hits / total,
            "average_savings_ratio": total_savings_ratio / total,
            "total_tokens_saved": total_tokens_saved,
            "slots_served_from_cache": slots_cache,
            "slots_served_from_inference": slots_inference,
        }

    def history(self) -> List[Dict]:
        """Return per-request token savings for graphing.

        Returns:
            List of dicts with request_number, prompt (truncated),
            tokens_saved, cache_hit, and savings_ratio.
        """
        out = []
        cumulative = 0
        for i, e in enumerate(self._entries, 1):
            saved = e.get("estimated_full_tokens", 0) - e.get("actual_tokens_used", 0)
            cumulative += saved
            out.append({
                "request_number": i,
                "prompt": (e.get("prompt", "") or "")[:50],
                "tokens_saved": saved,
                "cumulative_tokens_saved": cumulative,
                "cache_hit": e.get("cache_hit", False),
                "savings_ratio": e.get("savings_ratio", 0.0),
            })
        return out

