"""ClusterRouter — two-step routing via template clusters for fast lookup.

Groups intent centroids into semantic clusters. Routes queries by first
matching a cluster, then searching within that cluster for the best centroid.
Falls back to flat scan if cluster count is too low.
"""

from __future__ import annotations

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

# Domain keyword sets for tiebreaking and mismatch detection
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "space": [
        "planet", "planets", "solar", "star", "orbit", "moon", "nasa",
        "galaxy", "asteroid", "comet", "telescope", "jupiter", "saturn",
        "mars", "mercury", "venus", "neptune", "uranus", "solar system",
        "dwarf planet", "largest planet", "smallest planet",
    ],
    "geography": ["country", "city", "capital", "continent", "area",
                   "population", "border", "territory"],
    "finance": ["price", "cost", "dollar", "market", "stock", "revenue",
                "profit", "budget"],
    "biology": ["cell", "organism", "species", "gene", "protein",
                "evolution", "ecosystem"],
    "history": ["war", "century", "empire", "revolution", "treaty",
                "civilization"],
}

# Subdomain keyword sets for intra-domain tiebreaking
SUBDOMAIN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "space": {
        "planets": [
            "planet", "planets", "solar system", "jupiter", "saturn",
            "mars", "mercury", "venus", "earth", "neptune", "uranus",
            "orbit", "dwarf planet",
        ],
        "stars": [
            "star", "stars", "sun", "supernova", "hypergiant",
            "red giant", "white dwarf", "neutron star", "pulsar",
            "uy scuti",
        ],
        "galaxies": [
            "galaxy", "milky way", "andromeda", "nebula", "black hole",
            "quasar",
        ],
    },
    "geography": {
        "countries": [
            "country", "countries", "nation", "territory", "border",
            "capital",
        ],
        "cities": [
            "city", "cities", "urban", "metropolitan", "population",
        ],
    },
}

# Map from intent ID prefixes / cluster labels to domain names
_LABEL_TO_DOMAIN: Dict[str, str] = {
    "space": "space",
    "comp": "geography",
    "geo": "geography",
    "fin": "finance",
    "bio": "biology",
    "hist": "history",
}

# Similarity gap threshold that triggers tiebreaking
_TIEBREAK_GAP = 0.05

# Validate that critical keywords are present (catches stale bytecode)
assert "planet" in DOMAIN_KEYWORDS.get("space", []), \
    "DOMAIN_KEYWORDS['space'] missing 'planet' — check for stale bytecode"


def _domain_score(query_lower: str, domain: str) -> int:
    """Score how well a query matches a domain's keywords.

    Args:
        query_lower: Lowercased query text.
        domain: Domain name key in DOMAIN_KEYWORDS.

    Returns:
        Count of domain keywords found in the query.
    """
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    return sum(1 for kw in keywords if kw in query_lower)


def _intent_domain(intent_id: str, cluster_label: str | None) -> str | None:
    """Resolve the domain of an intent from its ID prefix or cluster label.

    Args:
        intent_id: The intent identifier (e.g. 'space_solar_system').
        cluster_label: The cluster label (e.g. 'space', 'comp').

    Returns:
        Domain name string or None if no domain can be resolved.
    """
    prefix = intent_id.split("_")[0] if intent_id else ""
    if prefix in _LABEL_TO_DOMAIN:
        return _LABEL_TO_DOMAIN[prefix]
    if cluster_label and cluster_label in _LABEL_TO_DOMAIN:
        return _LABEL_TO_DOMAIN[cluster_label]
    return None


def _domain_tiebreak(
    query: str, candidates: List[Tuple[float, "IntentCentroid", str | None]]
) -> Tuple[float, "IntentCentroid", str | None] | None:
    """Break ties between candidate centroids using domain keyword matching.

    When top candidates are within _TIEBREAK_GAP similarity of each other,
    uses domain keyword scores to select the better match. Also vetoes a
    top candidate when the query's domain keywords strongly disagree with
    the candidate's domain.

    Args:
        query: The user query text.
        candidates: List of (score, centroid, cluster_label) tuples,
            sorted descending by score.

    Returns:
        The winning (score, centroid, cluster_label) tuple, or None to
        let the caller use default behaviour.
    """
    if len(candidates) < 2:
        return None

    query_lower = query.lower()

    # Score the query against all domains
    query_domain_scores: Dict[str, int] = {}
    for domain in DOMAIN_KEYWORDS:
        query_domain_scores[domain] = _domain_score(query_lower, domain)

    # Find the query's best domain
    best_query_domain = max(query_domain_scores, key=query_domain_scores.get)
    best_query_domain_score = query_domain_scores[best_query_domain]

    # If query has no domain signal, can't tiebreak
    if best_query_domain_score == 0:
        return None

    top_score, top_centroid, top_label = candidates[0]
    second_score, second_centroid, second_label = candidates[1]

    top_domain = _intent_domain(top_centroid.intent_id, top_label)
    second_domain = _intent_domain(second_centroid.intent_id, second_label)

    # Case 1: Top two are within tiebreak gap — use domain keywords
    if top_score - second_score <= _TIEBREAK_GAP:
        top_ds = query_domain_scores.get(top_domain, 0) if top_domain else 0
        second_ds = query_domain_scores.get(second_domain, 0) if second_domain else 0
        if second_ds >= top_ds + 2:
            return candidates[1]

    # Case 2: Top candidate's domain mismatches query's strong domain signal.
    # Even if embedding similarity is high, domain keyword mismatch overrides.
    if top_domain and top_domain != best_query_domain and best_query_domain_score >= 1:
        top_ds = query_domain_scores.get(top_domain, 0) if top_domain else 0
        # Query has strong signal for a different domain than the top candidate
        if top_ds == 0 and best_query_domain_score >= 1:
            # Find all candidates in the query's domain
            domain_candidates = [
                (s, c, l) for s, c, l in candidates
                if _intent_domain(c.intent_id, l) == best_query_domain
                and s >= INTENT_SIMILARITY_THRESHOLD
            ]
            if domain_candidates:
                # Apply subdomain tiebreak within the matched domain
                subdomain_winner = _subdomain_tiebreak(
                    query, domain_candidates, best_query_domain
                )
                return subdomain_winner or domain_candidates[0]

    # Case 3: Multiple candidates in the same domain — subdomain tiebreak.
    # Use a lower threshold to include rescue candidates that may be the
    # correct subdomain match despite lower embedding similarity.
    _SUBDOMAIN_RESCUE_THRESHOLD = INTENT_SIMILARITY_THRESHOLD - 0.15
    if top_domain:
        same_domain = [
            (s, c, l) for s, c, l in candidates
            if _intent_domain(c.intent_id, l) == top_domain
            and s >= _SUBDOMAIN_RESCUE_THRESHOLD
        ]
        if len(same_domain) >= 2:
            subdomain_winner = _subdomain_tiebreak(query, same_domain, top_domain)
            if subdomain_winner is not None:
                return subdomain_winner

    return None


def _subdomain_tiebreak(
    query: str,
    candidates: List[Tuple[float, "IntentCentroid", str | None]],
    domain: str,
) -> Tuple[float, "IntentCentroid", str | None] | None:
    """Break ties between candidates in the same domain using subdomain keywords.

    Scores each candidate's intent ID against subdomain keyword sets to find
    the best match within a domain (e.g. planets vs stars within space).

    Args:
        query: The user query text.
        candidates: List of (score, centroid, cluster_label) tuples, all in
            the same domain, sorted descending by score.
        domain: The domain name (key in SUBDOMAIN_KEYWORDS).

    Returns:
        The winning (score, centroid, cluster_label) tuple, or None if
        subdomain scoring doesn't differentiate the candidates.
    """
    subdomains = SUBDOMAIN_KEYWORDS.get(domain)
    if not subdomains:
        return None

    query_lower = query.lower()

    # Score the query against each subdomain
    query_subdomain_scores: Dict[str, int] = {}
    for subdomain, keywords in subdomains.items():
        query_subdomain_scores[subdomain] = sum(
            1 for kw in keywords if kw in query_lower
        )

    # Find the query's best subdomain
    best_subdomain = max(query_subdomain_scores, key=query_subdomain_scores.get)
    best_subdomain_score = query_subdomain_scores[best_subdomain]

    if best_subdomain_score == 0:
        return None

    # Score each candidate's intent ID against subdomain keywords
    best_candidate = None
    best_candidate_score = -1

    for score, centroid, label in candidates:
        intent_spaced = centroid.intent_id.lower().replace("_", " ")
        intent_words = set(intent_spaced.split())
        cand_subdomain_score = 0
        for kw in subdomains.get(best_subdomain, []):
            if " " in kw:
                if kw in intent_spaced:
                    cand_subdomain_score += 1
            else:
                if kw in intent_words:
                    cand_subdomain_score += 1

        if cand_subdomain_score > best_candidate_score:
            best_candidate_score = cand_subdomain_score
            best_candidate = (score, centroid, label)
        elif cand_subdomain_score == best_candidate_score and best_candidate is not None:
            # Tied subdomain score — prefer higher embedding similarity
            if score > best_candidate[0]:
                best_candidate = (score, centroid, label)

    if best_candidate is not None and best_candidate_score > 0:
        return best_candidate

    return None


def _get_intent_domain(intent_id: str) -> str | None:
    """Resolve the domain of an intent from its ID.

    Checks intent ID prefix against _LABEL_TO_DOMAIN, then falls back
    to keyword matching against the intent ID string.

    Args:
        intent_id: The intent identifier.

    Returns:
        Domain name string or None.
    """
    prefix = intent_id.split("_")[0] if intent_id else ""
    if prefix in _LABEL_TO_DOMAIN:
        return _LABEL_TO_DOMAIN[prefix]
    intent_lower = intent_id.lower()
    if any(w in intent_lower for w in ["space", "planet", "star", "solar", "galaxy"]):
        return "space"
    if any(w in intent_lower for w in ["country", "city", "geo", "comp_largest"]):
        return "geography"
    return None


def _get_centroid_subdomain(intent_id: str, domain: str) -> str | None:
    """Determine which subdomain a centroid belongs to from its intent ID.

    Checks both the raw intent ID and a space-separated version so that
    multi-word keywords like "solar system" match intent IDs like
    "space_solar_system".

    Args:
        intent_id: The intent identifier.
        domain: The parent domain name.

    Returns:
        Subdomain name or None if no match.
    """
    if domain not in SUBDOMAIN_KEYWORDS:
        return None
    intent_lower = intent_id.lower()
    intent_spaced = intent_lower.replace("_", " ")
    intent_words = set(intent_spaced.split())
    best_subdomain = None
    best_score = 0
    for subdomain, keywords in SUBDOMAIN_KEYWORDS[domain].items():
        score = 0
        for kw in keywords:
            if " " in kw:
                if kw in intent_spaced:
                    score += 1
            else:
                if kw in intent_words:
                    score += 1
        if score > best_score:
            best_score = score
            best_subdomain = subdomain
    return best_subdomain


def _get_query_subdomain(query_lower: str, domain: str) -> str | None:
    """Determine which subdomain a query belongs to from its keywords.

    Args:
        query_lower: Lowercased query text.
        domain: The parent domain name.

    Returns:
        Subdomain name or None if no match.
    """
    if domain not in SUBDOMAIN_KEYWORDS:
        return None
    best_subdomain = None
    best_score = 0
    for subdomain, keywords in SUBDOMAIN_KEYWORDS[domain].items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_subdomain = subdomain
    return best_subdomain


def _get_query_domain(query_lower: str) -> str | None:
    """Determine the domain of a query from its keywords.

    Scores the query against all domain keyword sets and returns the
    domain with the highest keyword match count.

    Args:
        query_lower: Lowercased query text.

    Returns:
        Domain name string or None if no keywords match.
    """
    best_domain = None
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain if best_score > 0 else None


def _subdomain_rescue_scan(
    query: str,
    query_embedding: List[float],
    winning_intent_id: str,
    all_centroids: List["IntentCentroid"],
    cache_store: Optional["CacheStore"] = None,
) -> Tuple[Optional["IntentCentroid"], float]:
    """Scan all centroids for a better domain/subdomain match to rescue.

    Handles two cases:
    1. Cross-domain mismatch: winner is in the wrong domain entirely
       (e.g. geography winner for a space query).
    2. Same-domain subdomain mismatch: winner is in the right domain
       but wrong subdomain (e.g. stars winner for a planets query).

    Args:
        query: The user query text.
        query_embedding: Pre-computed query embedding.
        winning_intent_id: The current winner's intent ID.
        all_centroids: All centroids to scan.
        cache_store: Optional CacheStore for template content scoring.

    Returns:
        Tuple of (best_candidate, best_score) or (None, 0.0).
    """
    query_lower = query.lower()

    winning_domain = _get_intent_domain(winning_intent_id)
    query_domain = _get_query_domain(query_lower)

    if query_domain is None:
        return None, 0.0

    # If same domain, check subdomain match
    if winning_domain == query_domain:
        winning_subdomain = _get_centroid_subdomain(
            winning_intent_id, winning_domain
        )
        query_subdomain = _get_query_subdomain(query_lower, query_domain)
        if winning_subdomain == query_subdomain:
            return None, 0.0
    # Cross-domain: query_subdomain is determined from query_domain
    query_subdomain = _get_query_subdomain(query_lower, query_domain)

    rescue_floor = INTENT_SIMILARITY_THRESHOLD - 0.15

    best_candidate = None
    best_score = 0.0

    for centroid in all_centroids:
        if centroid.intent_id == winning_intent_id:
            continue

        candidate_domain = _get_intent_domain(centroid.intent_id)
        if candidate_domain != query_domain:
            continue

        # If we have a subdomain signal, filter by it
        if query_subdomain is not None:
            candidate_subdomain = _get_centroid_subdomain(
                centroid.intent_id, query_domain
            )
            if candidate_subdomain != query_subdomain:
                continue

        similarity = cosine_similarity(
            query_embedding, centroid.centroid_embedding
        )
        if similarity >= rescue_floor and similarity > best_score:
            best_score = similarity
            best_candidate = centroid

    return best_candidate, best_score


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

    def __init__(self, cache_store: Optional["CacheStore"] = None) -> None:
        """Initialize ClusterRouter with empty state.

        Args:
            cache_store: Optional CacheStore for template content scoring
                during subdomain rescue tiebreaking.
        """
        self._clusters: List[Cluster] = []
        self._centroid_map: Dict[str, IntentCentroid] = {}
        self._is_built = False
        self._cache_store = cache_store

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

        # Subdomain rescue threshold: candidates slightly below the main
        # threshold can be promoted if subdomain keywords strongly match.
        _SUBDOMAIN_RESCUE_THRESHOLD = INTENT_SIMILARITY_THRESHOLD - 0.15

        all_candidates: List[Tuple[float, IntentCentroid, str]] = []
        rescue_candidates: List[Tuple[float, IntentCentroid, str]] = []
        for _, cluster in top_clusters:
            for cid in cluster.centroid_ids:
                centroid = self._centroid_map.get(cid)
                if centroid is None:
                    continue
                score = cosine_similarity(query_embedding, centroid.centroid_embedding)
                if score >= INTENT_SIMILARITY_THRESHOLD:
                    all_candidates.append((score, centroid, cluster.label))
                elif score >= _SUBDOMAIN_RESCUE_THRESHOLD:
                    rescue_candidates.append((score, centroid, cluster.label))
                if score > best_score:
                    best_score = score
                    best_centroid = centroid
                    best_cluster_label = cluster.label

        if best_score >= INTENT_SIMILARITY_THRESHOLD and best_centroid is not None:
            all_centroid_list = list(self._centroid_map.values())
            winner_domain = _get_intent_domain(best_centroid.intent_id)
            query_domain = _get_query_domain(query.lower())

            # Cross-domain mismatch: unconditional rescue regardless of
            # score margin. This fires when the winner is in the wrong
            # domain entirely (e.g. geography winner for a space query).
            if (
                winner_domain != query_domain
                and query_domain is not None
            ):
                rescue, rescue_score = _subdomain_rescue_scan(
                    query, query_embedding, best_centroid.intent_id,
                    all_centroid_list, cache_store=self._cache_store,
                )
                if rescue is not None:
                    logger.info(
                        "Cross-domain rescue: %s overrides %s (score %.4f)",
                        rescue.intent_id, best_centroid.intent_id,
                        rescue_score,
                    )
                    best_centroid = rescue
                    best_score = rescue_score
            else:
                # Same domain: try domain tiebreak among cluster candidates,
                # then subdomain rescue scan for intra-domain correction.
                combined = all_candidates + rescue_candidates
                if len(combined) >= 2:
                    combined.sort(key=lambda x: x[0], reverse=True)
                    override = _domain_tiebreak(query, combined)
                    if override is not None:
                        best_score, best_centroid, best_cluster_label = (
                            override
                        )
                        logger.info(
                            "Domain tiebreak: %s overrides default",
                            best_centroid.intent_id,
                        )

                rescue, rescue_score = _subdomain_rescue_scan(
                    query, query_embedding, best_centroid.intent_id,
                    all_centroid_list, cache_store=self._cache_store,
                )
                if rescue is not None:
                    logger.info(
                        "Subdomain rescue: %s overrides %s (score %.4f)",
                        rescue.intent_id, best_centroid.intent_id,
                        rescue_score,
                    )
                    best_centroid = rescue
                    best_score = rescue_score

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
        all_candidates: List[Tuple[float, IntentCentroid, None]] = []
        rescue_candidates: List[Tuple[float, IntentCentroid, None]] = []
        _SUBDOMAIN_RESCUE_THRESHOLD = INTENT_SIMILARITY_THRESHOLD - 0.15

        for centroid in centroids:
            score = cosine_similarity(query_embedding, centroid.centroid_embedding)
            if score >= INTENT_SIMILARITY_THRESHOLD:
                all_candidates.append((score, centroid, None))
            elif score >= _SUBDOMAIN_RESCUE_THRESHOLD:
                rescue_candidates.append((score, centroid, None))
            if score > best_score:
                best_score = score
                best_centroid = centroid

        if best_score >= INTENT_SIMILARITY_THRESHOLD and best_centroid is not None:
            winner_domain = _get_intent_domain(best_centroid.intent_id)
            query_domain = _get_query_domain(query.lower())

            if (
                winner_domain != query_domain
                and query_domain is not None
            ):
                rescue, rescue_score = _subdomain_rescue_scan(
                    query, query_embedding, best_centroid.intent_id,
                    centroids, cache_store=self._cache_store,
                )
                if rescue is not None:
                    logger.info(
                        "Cross-domain rescue (flat): %s overrides %s",
                        rescue.intent_id, best_centroid.intent_id,
                    )
                    best_centroid = rescue
                    best_score = rescue_score
            else:
                combined = all_candidates + rescue_candidates
                if len(combined) >= 2:
                    combined.sort(key=lambda x: x[0], reverse=True)
                    override = _domain_tiebreak(query, combined)
                    if override is not None:
                        best_score, best_centroid, _ = override
                        logger.info(
                            "Domain tiebreak (flat): %s overrides default",
                            best_centroid.intent_id,
                        )

                rescue, rescue_score = _subdomain_rescue_scan(
                    query, query_embedding, best_centroid.intent_id,
                    centroids, cache_store=self._cache_store,
                )
                if rescue is not None:
                    logger.info(
                        "Subdomain rescue (flat): %s overrides %s",
                        rescue.intent_id, best_centroid.intent_id,
                    )
                    best_centroid = rescue
                    best_score = rescue_score

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
