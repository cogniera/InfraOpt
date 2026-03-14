"""ResponseTemplate dataclass for cached LLM response skeletons."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List


@dataclass
class ResponseTemplate:
    """Skeleton string with [slot_name] markers and metadata.

    Attributes:
        intent_id: The intent this template serves.
        skeleton: The response text with [slot_name] placeholders.
        slots: Ordered list of slot names present in the skeleton.
        dependency_graph: Maps each slot to a list of slots it depends on.
        variant: One of 'short', 'detailed', or 'list'.
        hit_count: Number of times this template has been served.
        created_at: Timestamp when the template was first created.
    """

    intent_id: str
    skeleton: str
    slots: List[str]
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    variant: str = "detailed"
    hit_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

