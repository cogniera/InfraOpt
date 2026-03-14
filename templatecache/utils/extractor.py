"""Template extraction and variant determination utilities."""

import re
from typing import Dict, List, Tuple

from templatecache.config import VARIANT_DETAILED_MIN_TOKENS, VARIANT_SHORT_MAX_TOKENS
from templatecache.utils.embedder import cosine_similarity, embed

_ELABORATION_KEYWORDS = {"explain", "detail", "describe", "how does", "walk me through"}
_LIST_KEYWORDS = {"list", "what are", "give me", "options", "compare"}


def determine_variant(query: str, response_token_count: int | None = None) -> str:
    """Determine the variant tag for a query.

    Uses query content and optional response token count to decide variant.

    Args:
        query: The user query text.
        response_token_count: Optional token count of the response.

    Returns:
        One of 'short', 'detailed', or 'list'.
    """
    query_lower = query.lower()
    words = query_lower.split()

    # Check keyword-based rules
    has_elaboration = any(kw in query_lower for kw in _ELABORATION_KEYWORDS)
    has_list = any(kw in query_lower for kw in _LIST_KEYWORDS)

    # Token count based rules
    if response_token_count is not None:
        if response_token_count < VARIANT_SHORT_MAX_TOKENS:
            if not has_elaboration and not has_list:
                return "short"
        if response_token_count > VARIANT_DETAILED_MIN_TOKENS:
            return "detailed"

    # Keyword-based rules
    if has_elaboration:
        return "detailed"
    if has_list:
        return "list"

    # Short query heuristic
    if len(words) < 8 and not has_elaboration:
        return "short"

    # Default: detailed wins
    return "detailed"


def extract_template(response: str) -> Tuple[str, List[str], Dict[str, List[str]]]:
    """Extract a template skeleton from an LLM response.

    Identifies variable portions of the response and replaces them with
    [slot_name] markers. Returns the skeleton, ordered slot list, and
    dependency graph.

    Args:
        response: The full LLM response text.

    Returns:
        Tuple of (skeleton, slots, dependency_graph) where:
        - skeleton: response text with [slot_name] placeholders
        - slots: ordered list of slot names
        - dependency_graph: maps each slot to its dependency slots
    """
    lines = response.strip().split("\n")
    skeleton_lines: List[str] = []
    slots: List[str] = []
    dependency_graph: Dict[str, List[str]] = {}
    slot_counter = 0

    for line in lines:
        # Identify segments that look like variable content:
        # - Quoted strings, specific names, numbers, technical terms
        processed_line = line

        # Replace specific data patterns with slot markers
        patterns = [
            (r'"[^"]{3,}"', "quoted_content"),         # quoted strings
            (r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', "date"),  # dates
            (r'\b\d+\.\d+%?\b', "number"),              # decimal numbers
            (r'\$[\d,.]+', "currency"),                  # currency amounts
        ]

        for pattern, slot_type in patterns:
            matches = list(re.finditer(pattern, processed_line))
            for match in reversed(matches):
                slot_name = f"{slot_type}_{slot_counter}"
                slot_counter += 1
                slots.append(slot_name)
                # Slots depend on previously found slots in the same line
                deps = [s for s in slots[:-1] if s in processed_line.replace(
                    match.group(), f"[{slot_name}]"
                )]
                dependency_graph[slot_name] = deps
                processed_line = (
                    processed_line[:match.start()]
                    + f"[{slot_name}]"
                    + processed_line[match.end():]
                )

        skeleton_lines.append(processed_line)

    skeleton = "\n".join(skeleton_lines)

    # If no slots were extracted, keep the full response as the skeleton
    # with no slots — the slot engine will return it as-is.
    return skeleton, slots, dependency_graph


# Splitter patterns for breaking queries into distinct aspects
_ASPECT_SPLITTERS = re.compile(
    r",\s*(?:and\s+)?|(?:\band\b\s+)|(?:\balso\b\s+)|(?:\bplus\b\s+)|(?:\bas well as\b\s+)",
    re.IGNORECASE,
)

# Patterns that indicate separate questions in a single prompt
_QUESTION_SPLITTERS = re.compile(
    r"[,?]\s*(?:and\s+)?|(?:\band\b\s+)|(?:\balso\b\s+)|(?:\bplus\b\s+)|(?:\bas well as\b\s+)",
    re.IGNORECASE,
)

# Question starters that hint at a distinct question
_QUESTION_STARTERS = re.compile(
    r"^(?:what|how|why|when|where|who|which|can you|could you|explain|describe|list|tell me|give me)",
    re.IGNORECASE,
)


def split_multi_query(query: str) -> List[str]:
    """Split a multi-topic query into individual sub-questions.

    Detects when a query contains multiple distinct questions separated by
    commas, 'and', question marks, etc. Returns the individual questions.
    Returns a single-element list if the query is about one topic.

    Args:
        query: The user query text.

    Returns:
        List of individual sub-question strings. Single-element list if
        the query cannot be meaningfully split.
    """
    # Split on question marks, commas, and conjunctions
    parts = _QUESTION_SPLITTERS.split(query.strip())
    parts = [p.strip().rstrip("?").strip() for p in parts if len(p.strip()) > 3]

    if len(parts) <= 1:
        return [query.strip()]

    # Filter out parts that are too generic to be standalone questions
    # e.g. "give me examples", "tell me more" — these modify the prior part
    # A valid sub-question must start with a question word AND contain a
    # specific topic (not just "give me examples" or "explain more")
    _GENERIC_TAILS = {"examples", "more", "details", "info", "some", "that"}
    topic_parts = []
    for p in parts:
        words = p.lower().split()
        if not _QUESTION_STARTERS.search(p):
            continue
        if len(words) < 3:
            continue
        # Reject if the non-starter words are all generic
        content_words = set(words) - {"what", "is", "how", "does", "do", "why",
                                       "when", "where", "who", "which", "can",
                                       "you", "could", "explain", "describe",
                                       "list", "tell", "me", "give", "a", "an",
                                       "the", "it", "about"}
        if content_words and not content_words.issubset(_GENERIC_TAILS):
            topic_parts.append(p)

    # Need at least 2 topic-bearing parts to consider it multi-query
    if len(topic_parts) < 2:
        return [query.strip()]

    # Check if parts are semantically distinct (different topics)
    from templatecache.utils.embedder import cosine_similarity, embed

    embeddings = [embed(p) for p in topic_parts]
    distinct_count = 0
    for i in range(len(embeddings)):
        is_distinct = True
        for j in range(len(embeddings)):
            if i == j:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > 0.45:
                is_distinct = False
                break
        if is_distinct:
            distinct_count += 1

    # If most parts are semantically distinct, treat as multi-query
    if distinct_count >= len(topic_parts) // 2 and len(topic_parts) >= 2:
        return topic_parts

    return [query.strip()]

# Minimum similarity for an aspect to be considered "covered" by the response
_GAP_COVERAGE_THRESHOLD = 0.45

# Minimum similarity between entire query and cached response for the
# response to be considered relevant. Below this, the query is asking
# about a different facet of the same topic (e.g. "weapons in WW2" vs
# a cached response about "causes of WW2").
_RESPONSE_RELEVANCE_THRESHOLD = 0.35


def detect_query_gaps(query: str, cached_response: str) -> List[str]:
    """Detect aspects of a query not covered by the cached response.

    For multi-aspect queries: splits into clauses and checks each against
    the response. For single-aspect queries: checks if the query is
    semantically relevant to the response — if not, the entire query is
    returned as a gap (the cached response doesn't answer the question).

    Args:
        query: The user query text.
        cached_response: The cached template response text.

    Returns:
        List of query aspect strings that are NOT covered by the cached
        response. Empty list if the response fully covers the query.
    """
    response_embedding = embed(cached_response[:500])  # cap for efficiency

    # Split query into aspects
    aspects = _ASPECT_SPLITTERS.split(query.strip())
    aspects = [a.strip() for a in aspects if len(a.strip()) > 3]

    # Single aspect: check if the response is relevant to the query at all
    if len(aspects) <= 1:
        query_embedding = embed(query)
        relevance = cosine_similarity(query_embedding, response_embedding)
        if relevance < _RESPONSE_RELEVANCE_THRESHOLD:
            return [query]
        return []

    gaps: List[str] = []

    for aspect in aspects:
        aspect_embedding = embed(aspect)
        similarity = cosine_similarity(aspect_embedding, response_embedding)
        if similarity < _GAP_COVERAGE_THRESHOLD:
            gaps.append(aspect)

    return gaps

