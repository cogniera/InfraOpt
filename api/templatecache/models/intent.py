"""IntentCentroid dataclass for intent routing."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class IntentCentroid:
    """Centroid embedding for an intent cluster.

    Keyed by intent:{intent_id} in Redis.

    Attributes:
        intent_id: Unique identifier for this intent.
        centroid_embedding: Average embedding vector for queries of this intent.
        template_id: ID of the associated ResponseTemplate.
        variant: One of 'short', 'detailed', or 'list'.
        query_count: Number of queries that have contributed to this centroid.
    """

    intent_id: str
    centroid_embedding: List[float] = field(default_factory=list)
    template_id: str = ""
    variant: str = "detailed"
    query_count: int = 0

