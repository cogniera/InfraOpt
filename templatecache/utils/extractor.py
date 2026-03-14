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

# Minimum similarity for an aspect to be considered "covered" by the response
_GAP_COVERAGE_THRESHOLD = 0.45


def detect_query_gaps(query: str, cached_response: str) -> List[str]:
    """Detect aspects of a query not covered by the cached response.

    Splits the query into semantic aspects (clauses separated by 'and',
    commas, etc.), embeds each aspect and the cached response, then returns
    aspects whose similarity to the response falls below the coverage
    threshold.

    Args:
        query: The user query text.
        cached_response: The cached template response text.

    Returns:
        List of query aspect strings that are NOT covered by the cached
        response. Empty list if the response fully covers the query.
    """
    # Split query into aspects
    aspects = _ASPECT_SPLITTERS.split(query.strip())
    aspects = [a.strip() for a in aspects if len(a.strip()) > 3]

    # If only one aspect, no gaps possible — the intent already matched
    if len(aspects) <= 1:
        return []

    response_embedding = embed(cached_response[:500])  # cap for efficiency
    gaps: List[str] = []

    for aspect in aspects:
        aspect_embedding = embed(aspect)
        similarity = cosine_similarity(aspect_embedding, response_embedding)
        if similarity < _GAP_COVERAGE_THRESHOLD:
            gaps.append(aspect)

    return gaps

