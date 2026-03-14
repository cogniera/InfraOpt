"""SlotRecord dataclass for cached slot fill values."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import List


@dataclass
class SlotRecord:
    """A cached fill value for a specific slot in a specific context.

    Keyed by slot:{slot_id}:{context_hash} in Redis.

    Attributes:
        slot_id: Name of the slot this record fills.
        context_hash: Hash of the query context that produced this fill.
        fill_value: The actual text value to insert into the template.
        fill_embedding: Embedding vector of the fill value.
        similarity_score: Confidence score for this fill (0.0 to 1.0).
        decay_weight: Current decay multiplier applied to the score.
        created_at: Timestamp when this record was created.
    """

    slot_id: str
    context_hash: str
    fill_value: str
    fill_embedding: List[float] = field(default_factory=list)
    similarity_score: float = 0.0
    decay_weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

