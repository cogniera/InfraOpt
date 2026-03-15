"""IntentRouter — routes queries to cached templates by intent similarity."""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from templatecache.config import INTENT_SIMILARITY_THRESHOLD
from templatecache.models.intent import IntentCentroid
from templatecache.models.template import ResponseTemplate
from templatecache.modules.cache_store import CacheStore
from templatecache.utils.embedder import batch_embed, cosine_similarity, embed
from templatecache.utils.extractor import determine_variant, extract_template
from templatecache.utils.llm import llm_call

logger = logging.getLogger(__name__)


class IntentRouter:
    """Routes queries to cached templates by comparing against intent centroids.

    Answers one question: does a reusable template exist for this query?
    """

    def __init__(self, cache_store: CacheStore) -> None:
        """Initialize IntentRouter.

        Args:
            cache_store: CacheStore instance for Redis access.
        """
        self._cache_store = cache_store

    def route(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Check if a cached template exists for the given query.

        Embeds the query, compares against all intent centroid embeddings,
        returns an intent ID and variant tag if similarity exceeds threshold.

        Args:
            query: The user query text.

        Returns:
            Tuple of (intent_id, variant) if a match is found above threshold,
            or (None, None) if no match.

        Side effects:
            Calls the embedding API via embedder.embed().
        """
        query_embedding = embed(query)
        centroids = self._cache_store.get_all_intent_centroids()

        if not centroids:
            return None, None

        best_score = -1.0
        best_centroid: Optional[IntentCentroid] = None

        for centroid in centroids:
            score = cosine_similarity(query_embedding, centroid.centroid_embedding)
            if score > best_score:
                best_score = score
                best_centroid = centroid

        if best_score >= INTENT_SIMILARITY_THRESHOLD and best_centroid is not None:
            # Determine variant from query and check agreement with centroid
            query_variant = determine_variant(query)
            variant = best_centroid.variant
            # Both must agree or the detailed variant wins
            if query_variant != variant:
                variant = "detailed"
            return best_centroid.intent_id, variant

        return None, None

    async def seed_centroids(
        self, examples: List[Dict[str, str]]
    ) -> None:
        """Seed intent centroids from example query-response pairs.

        Called on first startup if no centroids exist. Blocks until complete.

        Args:
            examples: List of dicts with 'query', 'response', and 'intent_id' keys.
                Must have at least 5 examples covering at least 3 intent types and
                include at least one example of each variant (short, detailed, list).

        Side effects:
            Writes IntentCentroids and ResponseTemplates to Redis via CacheStore.
            Calls embedding API.
        """
        # Group examples by intent_id
        intent_groups: Dict[str, List[Dict[str, str]]] = {}
        for ex in examples:
            iid = ex["intent_id"]
            if iid not in intent_groups:
                intent_groups[iid] = []
            intent_groups[iid].append(ex)

        import numpy as np

        total = len(intent_groups)
        for i, (intent_id, group) in enumerate(intent_groups.items(), 1):
            # Embed all queries in this group
            queries = [ex["query"] for ex in group]
            embeddings = batch_embed(queries)

            # Compute centroid as average embedding
            centroid_vec = np.mean(embeddings, axis=0).tolist()

            # Use first example to build template
            first_response = group[0]["response"]
            variant = determine_variant(group[0]["query"])
            skeleton, slots, dep_graph, _slot_types, templateable = extract_template(
                first_response
            )

            template = ResponseTemplate(
                intent_id=intent_id,
                skeleton=skeleton,
                slots=slots,
                dependency_graph=dep_graph,
                variant=variant,
                templateable=templateable,
                raw_response=first_response if not templateable else "",
            )

            centroid = IntentCentroid(
                intent_id=intent_id,
                centroid_embedding=centroid_vec,
                template_id=intent_id,
                variant=variant,
                query_count=len(group),
            )

            await self._cache_store.write_back(template=template, centroid=centroid)

            # Progress logging every 25 intents or on the last one
            if i % 25 == 0 or i == total:
                print(f"  Seeded {i}/{total} intents...", flush=True)

        logger.info("Seeded %d intent centroids", total)

