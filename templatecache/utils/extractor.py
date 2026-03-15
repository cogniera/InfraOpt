"""Template extraction and variant determination utilities."""

import re
from typing import Dict, List, Tuple

from templatecache.config import VARIANT_DETAILED_MIN_TOKENS, VARIANT_SHORT_MAX_TOKENS
from templatecache.utils.embedder import cosine_similarity, embed

_ELABORATION_KEYWORDS = {"explain", "detail", "describe", "how does", "walk me through"}
_LIST_KEYWORDS = {"list", "what are", "give me", "options", "compare"}

# ── Slot type classification ─────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"^\$[\d,.]+$|^[\d,.]+\s*(USD|EUR|GBP|JPY|CAD|AUD)$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
_PERCENTAGE_RE = re.compile(r"^[\d.]+%$")
_NUMERIC_RE = re.compile(r"^[\d,.]+$")
_NAMED_ENTITY_RE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$")

# Duration patterns — time spans like "5-7 business days", "within 24 hours"
_DURATION_RE_RANGE = re.compile(
    r"\d+[-–]\d+\s*(days?|hours?|weeks?|months?|business\s+days?|working\s+days?)",
    re.IGNORECASE,
)
_DURATION_RE_WITHIN = re.compile(
    r"within\s+\d+\s*(days?|hours?|weeks?|months?)", re.IGNORECASE
)
_DURATION_RE_BUSINESS = re.compile(
    r"\d+\s*(business\s+days?|working\s+days?)", re.IGNORECASE
)

# Maximum number of slots to extract from a single response
_MAX_SLOT_COUNT = 6


def classify_slot(value: str) -> str:
    """Classify a slot fill value into a semantic type.

    Classification order (first match wins):
    1. duration — time spans (ranges, "within X days", business days)
    2. currency — dollar signs or currency codes
    3. date — date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
    4. percentage — number followed by %
    5. named_entity — two or more capitalized words (proper nouns)
    6. numeric — plain numbers with optional commas/decimals
    7. boilerplate — everything else (stable, free text)

    Args:
        value: The slot fill value string to classify.

    Returns:
        One of: 'duration', 'currency', 'date', 'percentage',
        'named_entity', 'numeric', 'boilerplate'.
    """
    v = value.strip()
    if not v:
        return "boilerplate"
    if _DURATION_RE_RANGE.search(v):
        return "duration"
    if _DURATION_RE_WITHIN.search(v):
        return "duration"
    if _DURATION_RE_BUSINESS.search(v):
        return "duration"
    if _CURRENCY_RE.match(v):
        return "currency"
    if _DATE_RE.match(v):
        return "date"
    if _PERCENTAGE_RE.match(v):
        return "percentage"
    if _NAMED_ENTITY_RE.match(v):
        return "named_entity"
    if _NUMERIC_RE.match(v):
        return "numeric"
    return "boilerplate"


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


# Words to skip when deriving semantic labels from context
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "it", "its", "this", "that",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "about", "than", "over", "under",
})


def _derive_semantic_label(
    line: str, match_start: int, slot_type: str, used_labels: set
) -> str:
    """Derive a descriptive slot name from surrounding context.

    Looks at the 1-3 words immediately before the matched value in the
    line to create a meaningful label like 'contribution_limit' instead
    of 'currency_0'.

    Args:
        line: The current line being processed.
        match_start: Character index where the matched value starts.
        slot_type: The regex-derived type (e.g. 'currency', 'number').
        used_labels: Set of labels already used (to avoid duplicates).

    Returns:
        A descriptive slot name like 'contribution_limit_currency' or
        falls back to '{slot_type}_{n}' if no context is available.
    """
    # Get text before the match, extract last 1-3 meaningful words
    prefix = line[:match_start].strip().rstrip(":(-–—")
    words = [w.lower().strip(".,;:()") for w in prefix.split()]
    # Filter out stop words and slot markers
    context_words = [
        w for w in words
        if w and w not in _STOP_WORDS and not w.startswith("[")
    ]

    if context_words:
        # Take last 1-2 context words as the label
        label_parts = context_words[-2:] if len(context_words) >= 2 else context_words[-1:]
        # Clean: only keep alphanumeric and underscores
        label = "_".join(re.sub(r"[^a-z0-9]", "", p) for p in label_parts if re.sub(r"[^a-z0-9]", "", p))
        if label:
            base = f"{label}_{slot_type}"
        else:
            base = slot_type
    else:
        base = slot_type

    # Ensure uniqueness
    candidate = base
    counter = 0
    while candidate in used_labels:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


def _calculate_code_block_ratio(response: str) -> float:
    """Calculate the fraction of response content inside code blocks.

    Args:
        response: The full LLM response text.

    Returns:
        Float between 0.0 and 1.0 representing the proportion of
        tokens inside backtick-fenced code blocks.
    """
    total_tokens = len(response.split())
    if total_tokens == 0:
        return 0.0

    code_tokens = 0
    in_code = False
    for line in response.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_tokens += len(line.split())

    return code_tokens / total_tokens


def extract_template(
    response: str,
) -> Tuple[str, List[str], Dict[str, List[str]], Dict[str, str], bool]:
    """Extract a template skeleton from an LLM response.

    Identifies variable portions of the response and replaces them with
    [slot_name] markers. Returns the skeleton, ordered slot list,
    dependency graph, slot type classifications, and whether the response
    is suitable for templating.

    A response is not templateable if more than 60% of its content is
    inside backtick-fenced code blocks.

    After extraction, if more than 6 slots are found, only the 6 with
    the longest fill values are kept — the rest are treated as literal
    skeleton text.

    Args:
        response: The full LLM response text.

    Returns:
        Tuple of (skeleton, slots, dependency_graph, slot_types,
        templateable) where:
        - skeleton: response text with [slot_name] placeholders
        - slots: ordered list of slot names
        - dependency_graph: maps each slot to its dependency slots
        - slot_types: maps each slot name to its classified type
        - templateable: False if response is >60% code blocks
    """
    # ── Templateable check ────────────────────────────────────────────
    code_ratio = _calculate_code_block_ratio(response)
    if code_ratio > 0.60:
        return response, [], {}, {}, False

    # ── Slot extraction ───────────────────────────────────────────────
    lines = response.strip().split("\n")
    skeleton_lines: List[str] = []
    slots: List[str] = []
    slot_fills: Dict[str, str] = {}  # slot_name → original matched text
    dependency_graph: Dict[str, List[str]] = {}
    slot_types: Dict[str, str] = {}
    slot_counter = 0
    used_labels: set = set()
    in_code_block = False

    for line in lines:
        # Track fenced code blocks — don't extract slots from code
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            skeleton_lines.append(line)
            continue

        if in_code_block:
            skeleton_lines.append(line)
            continue

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
                # Derive a semantic label from surrounding context
                label = _derive_semantic_label(
                    processed_line, match.start(), slot_type, used_labels
                )
                slot_name = label
                used_labels.add(label)
                slot_counter += 1
                slots.append(slot_name)
                slot_fills[slot_name] = match.group()
                # Classify the matched value
                slot_types[slot_name] = classify_slot(match.group())
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

    # ── Slot count cap ────────────────────────────────────────────────
    # If more than _MAX_SLOT_COUNT slots were extracted, keep only the
    # ones with the longest fill values (most likely genuinely variable).
    # Discard the rest by replacing their markers back with literal text.
    if len(slots) > _MAX_SLOT_COUNT:
        # Sort by fill value length descending, keep top _MAX_SLOT_COUNT
        ranked = sorted(slots, key=lambda s: len(slot_fills.get(s, "")), reverse=True)
        keep = set(ranked[:_MAX_SLOT_COUNT])
        discard = [s for s in slots if s not in keep]

        for slot_name in discard:
            # Replace [slot_name] marker back with original text
            skeleton = skeleton.replace(f"[{slot_name}]", slot_fills[slot_name])
            del dependency_graph[slot_name]
            del slot_types[slot_name]
            # Remove discarded slots from other slots' dependency lists
            for dep_list in dependency_graph.values():
                if slot_name in dep_list:
                    dep_list.remove(slot_name)

        slots = [s for s in slots if s in keep]

    return skeleton, slots, dependency_graph, slot_types, True


# Splitter patterns for breaking queries into distinct aspects
_ASPECT_SPLITTERS = re.compile(
    r",\s*(?:and\s+)?|(?:\band\b\s+)|(?:\balso\b\s+)|(?:\bplus\b\s+)"
    r"|(?:\bas well as\b\s+)|(?:\bbut\b\s+)|(?:\bhowever\b\s+)"
    r"|(?:\bwhat about\b\s+)|(?:\bhow about\b\s+)|(?:\bthough\b\s+)",
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

from templatecache.config import GAP_COVERAGE_THRESHOLD, RESPONSE_RELEVANCE_THRESHOLD


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
        if relevance < RESPONSE_RELEVANCE_THRESHOLD:
            return [query]
        return []

    gaps: List[str] = []

    for aspect in aspects:
        aspect_embedding = embed(aspect)
        similarity = cosine_similarity(aspect_embedding, response_embedding)
        if similarity < GAP_COVERAGE_THRESHOLD:
            gaps.append(aspect)

    return gaps

