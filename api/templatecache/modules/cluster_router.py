"""ClusterRouter — two-step routing via template clusters for fast lookup.

Groups intent centroids into semantic clusters. Routes queries by first
matching a cluster, then searching within that cluster for the best centroid.
Falls back to flat scan if cluster count is too low.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from templatecache.config import INTENT_SIMILARITY_THRESHOLD
from templatecache.models.intent import IntentCentroid
from templatecache.utils.embedder import cosine_similarity, embed
from templatecache.utils.extractor import determine_variant

logger = logging.getLogger(__name__)

# Minimum centroids to justify clustering; below this, flat scan is faster
MIN_CENTROIDS_FOR_CLUSTERING = 50

# How many top clusters to search (in case query sits on a cluster boundary)
TOP_K_CLUSTERS = 2


@dataclass
class Cluster:
    """A group of semantically related intent centroids.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        center: Mean embedding vector of all centroids in this cluster.
        centroid_ids: Intent IDs of centroids belonging to this cluster.
        label: Human-readable label (derived from most common intent prefix).
    """

    cluster_id: int
    center: List[float] = field(default_factory=list)
    centroid_ids: List[str] = field(default_factory=list)
    label: str = ""


class ClusterRouter:
    """Two-step router: cluster match → centroid match within cluster.

    Reduces comparisons from O(n) to O(k + n/k) where k = cluster count.
    Falls back to flat scan when centroid count is below MIN_CENTROIDS_FOR_CLUSTERING.
    """

    def __init__(self) -> None:
        """Initialize ClusterRouter with empty state."""
        self._clusters: List[Cluster] = []
        self._centroid_map: Dict[str, IntentCentroid] = {}
        self._is_built = False

    @property
    def cluster_count(self) -> int:
        """Return number of clusters."""
        return len(self._clusters)

    @property
    def is_built(self) -> bool:
        """Return whether clusters have been built."""
        return self._is_built

    def build(self, centroids: List[IntentCentroid], n_clusters: Optional[int] = None) -> None:
        """Build clusters from a list of intent centroids using k-means.

        Args:
            centroids: All intent centroids from Redis.
            n_clusters: Number of clusters. Auto-calculated if None.

        Side effects:
            Populates self._clusters and self._centroid_map.
        """
        if len(centroids) < MIN_CENTROIDS_FOR_CLUSTERING:
            logger.info(
                "Only %d centroids, below threshold %d — skipping clustering",
                len(centroids), MIN_CENTROIDS_FOR_CLUSTERING,
            )
            self._centroid_map = {c.intent_id: c for c in centroids}
            self._is_built = False
            return

        # Auto-calculate cluster count: sqrt(n) is a good heuristic
        if n_clusters is None:
            n_clusters = max(5, int(math.sqrt(len(centroids))))

        self._centroid_map = {c.intent_id: c for c in centroids}
        embeddings = np.array([c.centroid_embedding for c in centroids])

        # Simple k-means (no sklearn dependency)
        assignments = self._kmeans(embeddings, n_clusters, max_iter=20)

        # Build cluster objects
        self._clusters = []
        for cluster_id in range(n_clusters):
            member_indices = [i for i, a in enumerate(assignments) if a == cluster_id]
            if not member_indices:
                continue

            member_embeddings = embeddings[member_indices]
            center = np.mean(member_embeddings, axis=0).tolist()
            centroid_ids = [centroids[i].intent_id for i in member_indices]

            # Derive label from most common intent prefix
            prefixes = [cid.split("_")[0] for cid in centroid_ids]
            label = max(set(prefixes), key=prefixes.count)

            self._clusters.append(Cluster(
                cluster_id=cluster_id,
                center=center,
                centroid_ids=centroid_ids,
                label=label,
            ))

        self._is_built = True
        logger.info(
            "Built %d clusters from %d centroids (avg %.1f per cluster)",
            len(self._clusters), len(centroids),
            len(centroids) / max(len(self._clusters), 1),
        )



    def _kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> List[int]:
        """Run simple k-means clustering.

        Args:
            data: Array of shape (n, dim) — embedding vectors.
            k: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            List of cluster assignments, one per data point.
        """
        n = len(data)
        # Initialize centers using k-means++ style: spread out initial picks
        rng = np.random.RandomState(42)
        center_indices = [rng.randint(n)]
        for _ in range(1, k):
            # Pick next center with probability proportional to distance
            dists = np.array([
                min(np.dot(data[i] - data[c], data[i] - data[c]) for c in center_indices)
                for i in range(n)
            ])
            dists /= dists.sum() + 1e-10
            center_indices.append(rng.choice(n, p=dists))

        centers = data[center_indices].copy()
        assignments = [0] * n

        for _ in range(max_iter):
            # Assign each point to nearest center using cosine similarity
            new_assignments = []
            for i in range(n):
                best_cluster = 0
                best_sim = -1.0
                for j in range(len(centers)):
                    sim = float(np.dot(data[i], centers[j]) / (
                        np.linalg.norm(data[i]) * np.linalg.norm(centers[j]) + 1e-10
                    ))
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = j
                new_assignments.append(best_cluster)

            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Recompute centers
            for j in range(len(centers)):
                members = [i for i, a in enumerate(assignments) if a == j]
                if members:
                    centers[j] = np.mean(data[members], axis=0)

        return assignments

    def route(
        self, query: str, all_centroids: Optional[List[IntentCentroid]] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Route a query using two-step cluster matching.

        Step 1: Find the top-k closest clusters by cosine similarity.
        Step 2: Search only centroids within those clusters.

        Falls back to flat scan if clusters aren't built.

        Args:
            query: The user query text.
            all_centroids: All centroids (used for flat scan fallback).

        Returns:
            Tuple of (intent_id, variant, cluster_label) if match found,
            or (None, None, None) if no match.

        Side effects:
            Calls embed() for the query embedding.
        """
        query_embedding = embed(query)

        # Flat scan fallback
        if not self._is_built:
            centroids = all_centroids or list(self._centroid_map.values())
            if not centroids:
                return None, None, None
            return self._flat_scan(query, query_embedding, centroids)

        # Step 1: Find top-k clusters
        cluster_scores = []
        for cluster in self._clusters:
            sim = cosine_similarity(query_embedding, cluster.center)
            cluster_scores.append((sim, cluster))
        cluster_scores.sort(key=lambda x: x[0], reverse=True)

        top_clusters = cluster_scores[:TOP_K_CLUSTERS]

        # Step 2: Search centroids within top clusters
        best_score = -1.0
        best_centroid: Optional[IntentCentroid] = None
        best_cluster_label: Optional[str] = None

        for _, cluster in top_clusters:
            for cid in cluster.centroid_ids:
                centroid = self._centroid_map.get(cid)
                if centroid is None:
                    continue
                score = cosine_similarity(query_embedding, centroid.centroid_embedding)
                if score > best_score:
                    best_score = score
                    best_centroid = centroid
                    best_cluster_label = cluster.label

        if best_score >= INTENT_SIMILARITY_THRESHOLD and best_centroid is not None:
            query_variant = determine_variant(query)
            variant = best_centroid.variant
            if query_variant != variant:
                variant = "detailed"
            return best_centroid.intent_id, variant, best_cluster_label

        # Cluster search missed — fall back to full scan in case the
        # centroid is in a cluster we didn't check (e.g. newly added)
        all_centroids = list(self._centroid_map.values())
        if all_centroids:
            result = self._flat_scan(query, query_embedding, all_centroids)
            if result[0] is not None:
                logger.info(
                    "Cluster miss but flat scan hit: %s", result[0]
                )
            return result

        return None, None, None

    def _flat_scan(
        self,
        query: str,
        query_embedding: List[float],
        centroids: List[IntentCentroid],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Flat scan fallback when clusters aren't built.

        Args:
            query: The user query text.
            query_embedding: Pre-computed query embedding.
            centroids: All centroids to scan.

        Returns:
            Tuple of (intent_id, variant, cluster_label=None).
        """
        best_score = -1.0
        best_centroid: Optional[IntentCentroid] = None

        for centroid in centroids:
            score = cosine_similarity(query_embedding, centroid.centroid_embedding)
            if score > best_score:
                best_score = score
                best_centroid = centroid

        if best_score >= INTENT_SIMILARITY_THRESHOLD and best_centroid is not None:
            query_variant = determine_variant(query)
            variant = best_centroid.variant
            if query_variant != variant:
                variant = "detailed"
            return best_centroid.intent_id, variant, None

        return None, None, None

    def update_centroid(self, centroid: IntentCentroid) -> None:
        """Update or add a centroid in the in-memory map.

        If the centroid already exists, updates it in place. If it's new,
        adds it to the map and assigns it to the nearest cluster.

        Args:
            centroid: The IntentCentroid to update or add.

        Side effects:
            Updates self._centroid_map. May add to a cluster's centroid_ids.
        """
        is_new = centroid.intent_id not in self._centroid_map
        self._centroid_map[centroid.intent_id] = centroid

        # For new centroids, assign to the nearest cluster
        if is_new and self._is_built and self._clusters:
            best_sim = -1.0
            best_cluster = self._clusters[0]
            for cluster in self._clusters:
                sim = cosine_similarity(centroid.centroid_embedding, cluster.center)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster
            best_cluster.centroid_ids.append(centroid.intent_id)
            logger.info(
                "Added new centroid '%s' to cluster '%s' (%d members)",
                centroid.intent_id, best_cluster.label,
                len(best_cluster.centroid_ids),
            )

    def get_cluster_info(self) -> List[Dict]:
        """Return summary info about all clusters.

        Returns:
            List of dicts with cluster_id, label, size, and sample intent IDs.
        """
        return [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "size": len(c.centroid_ids),
                "sample_intents": c.centroid_ids[:5],
            }
            for c in self._clusters
        ]
