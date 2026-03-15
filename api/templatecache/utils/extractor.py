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
# Named entities: two+ words with at least two capitalized, allows lowercase
# particles like "van", "de", "von", "di", "del", "la" between them.
_NAMED_ENTITY_RE = re.compile(r"^[A-Z][a-z]+(?: (?:[a-z]{1,4} )*[A-Z][a-z]+)+$")

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


# ── Answer extraction from list-style responses ─────────────────────────

# Superlative/comparative patterns that signal a specific-item query
_SPECIFIC_QUERY_RE = re.compile(
    r"\b((?:the\s+)?(?:largest|biggest|smallest|fastest|slowest|hottest|coldest|"
    r"tallest|shortest|heaviest|lightest|oldest|newest|closest|farthest|nearest|"
    r"most\s+\w+|least\s+\w+|best|worst|cheapest|most expensive|"
    r"first|last|longest|deepest|highest|lowest))\b",
    re.IGNORECASE,
)

# Pattern to extract "Name (descriptor, ...)" items from a list response
_LIST_ITEM_RE = re.compile(
    r"([A-Z][a-zA-Z\s'-]+?)\s*\(([^)]+)\)"
)


def _format_compound_list_response(query: str, response: str) -> str | None:
    """Format a compound query response with a direct answer prepended.

    When a query contains both a superlative signal and a list signal
    joined by a compound conjunction, extracts the specific answer for
    the superlative part and prepends it before the full list.

    Args:
        query: The user query text.
        response: The full cached list response text.

    Returns:
        Formatted string with direct answer followed by full list,
        or None if the query is not a compound superlative+list query.
    """
    query_lower = query.lower()

    SUPERLATIVE_SIGNALS = [
        "largest", "smallest", "biggest", "fastest", "slowest",
        "highest", "lowest", "cheapest", "oldest", "newest",
        "first", "last", "closest", "furthest", "heaviest",
        "lightest", "hottest", "coldest", "brightest",
    ]

    LIST_SIGNALS = [
        "list", "rest", "all", "every", "others", "remaining",
        "the rest", "list the", "list all",
    ]

    COMPOUND_SIGNALS = ["then", "and also", "as well as", "and then", "plus"]

    has_superlative = any(s in query_lower for s in SUPERLATIVE_SIGNALS)
    has_list = any(s in query_lower for s in LIST_SIGNALS)
    has_compound = any(s in query_lower for s in COMPOUND_SIGNALS)

    if not (has_superlative and has_list and has_compound):
        return None

    # Strip the compound part so extract_specific_answer sees a clean
    # single-fact query that passes gate 2 and gate 3.
    specific_part = query.split("then")[0].split("and also")[0].strip()

    extracted = extract_specific_answer(specific_part, response)

    if extracted is None:
        return None

    return f"{extracted}\n\nFull list:\n{response}"


def extract_specific_answer(query: str, response: str) -> str | None:
    """Extract a specific answer from a list-style cached response.

    When the query asks for a specific item (e.g. "what's the largest
    planet") but the cached response is a full list, this function
    finds the matching item and returns a focused answer.

    Args:
        query: The user query text.
        response: The full cached response text.

    Returns:
        A focused answer string if extraction succeeds, or None if
        the query isn't asking for a specific item or no match is found.
    """
    from templatecache.config import ANSWER_EXTRACTION_MIN_SCORE

    query_lower = query.lower()

    # ── Gate 1: Superlative signal ────────────────────────────────────
    SUPERLATIVE_SIGNALS = [
        "largest", "smallest", "biggest", "fastest", "slowest",
        "highest", "lowest", "cheapest", "most expensive", "oldest",
        "newest", "first", "last", "closest", "furthest",
        "heaviest", "lightest", "hottest", "coldest", "brightest",
    ]
    has_superlative = any(signal in query_lower for signal in SUPERLATIVE_SIGNALS)
    if not has_superlative:
        return None

    # ── Gate 2: Specific question starter ─────────────────────────────
    SPECIFIC_QUESTION_STARTERS = [
        "which", "what is the", "what's the", "who is the",
        "who's the", "name the", "tell me the", "what was the",
    ]
    has_question_starter = any(
        starter in query_lower for starter in SPECIFIC_QUESTION_STARTERS
    )
    if not has_question_starter:
        return None

    # ── Gate 3: Compound query abort ──────────────────────────────────
    COMPOUND_SIGNALS = [
        "then", "and also", "as well as", "plus", "along with", "and then",
    ]
    has_compound = any(signal in query_lower for signal in COMPOUND_SIGNALS)
    if has_compound:
        return None

    # ── Gate 4: List intent abort ─────────────────────────────────────
    _LIST_INTENT_RE = re.compile(
        r"\b(list|name all|all of|all the|every|each|enumerate)\b",
        re.IGNORECASE,
    )
    if _LIST_INTENT_RE.search(query):
        return None

    # ── Parse list items ──────────────────────────────────────────────
    items = _LIST_ITEM_RE.findall(response)
    if not items:
        return None

    # Synonym mapping for common superlatives
    _SYNONYMS = {
        "biggest": "largest",
        "largest": "biggest",
        "smallest": "tiniest",
        "tiniest": "smallest",
        "quickest": "fastest",
        "fastest": "quickest",
        "furthest": "farthest",
        "farthest": "furthest",
    }

    # ── Find the superlative adjective ────────────────────────────────
    match = _SPECIFIC_QUERY_RE.search(query)
    if not match:
        return None

    adjective = match.group(1).lower().strip()
    adjective = re.sub(r"^the\s+", "", adjective)
    adjective_alt = _SYNONYMS.get(adjective)

    candidates = [adjective]
    if adjective_alt:
        candidates.append(adjective_alt)

    # ── Match against descriptors with scoring ────────────────────────
    for idx, (name, descriptors) in enumerate(items):
        desc_lower = descriptors.lower()
        for adj in candidates:
            if adj in desc_lower:
                name = name.strip()

                # Score the match
                score = 0.0
                # +2 for each criterion keyword found in descriptors
                for signal in SUPERLATIVE_SIGNALS:
                    if signal in desc_lower:
                        score += 2.0
                # +3 for positional correctness
                if adjective in ("first", "closest", "nearest") and idx == 0:
                    score += 3.0
                elif adjective in ("last", "furthest", "farthest") and idx == len(items) - 1:
                    score += 3.0
                elif adjective not in ("first", "last", "closest", "nearest", "furthest", "farthest"):
                    # Non-positional superlative: +3 for direct descriptor match
                    score += 3.0

                if score < ANSWER_EXTRACTION_MIN_SCORE:
                    return None

                return f"{name}. It is the {adjective} ({descriptors.strip()})."

    # Check for "most/least X" patterns
    adj_core = re.sub(r"^(most|least)\s+", "", adjective)
    if adj_core != adjective:
        for idx, (name, descriptors) in enumerate(items):
            if adj_core in descriptors.lower():
                name = name.strip()
                score = 2.0 + 3.0  # keyword match + non-positional
                if score < ANSWER_EXTRACTION_MIN_SCORE:
                    return None
                return f"{name}. It is the {adjective} ({descriptors.strip()})."

    return None


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
        # Clean: only keep lowercase letters and underscores
        label = "_".join(re.sub(r"[^a-z]", "", p) for p in label_parts if re.sub(r"[^a-z]", "", p))
        if label:
            base = f"{label}_{slot_type}"
        else:
            base = slot_type
    else:
        base = slot_type

    # Ensure label starts with a letter (never a digit)
    if base and not base[0].isalpha():
        base = f"{slot_type}_{base}"

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
        return "", [], {}, {}, False

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
    r"|(?:\bwhat about\b\s+)|(?:\bhow about\b\s+)|(?:\bthough\b\s+)"
    r"|(?:\bthen\b\s*)|(?:\bafter that\b\s+)|(?:\bfollowed by\b\s+)"
    r"|(?:\band then\b\s+)|(?:\bnext\b\s+)",
    re.IGNORECASE,
)

# Patterns that indicate separate questions in a single prompt
_QUESTION_SPLITTERS = re.compile(
    r"[,?]\s*(?:and\s+)?|(?:\band\b\s+)|(?:\balso\b\s+)|(?:\bplus\b\s+)"
    r"|(?:\bas well as\b\s+)|(?:\bthen\b\s*)|(?:\bafter that\b\s+)"
    r"|(?:\bfollowed by\b\s+)|(?:\band then\b\s+)",
    re.IGNORECASE,
)

# Question starters that hint at a distinct question
_QUESTION_STARTERS = re.compile(
    r"^(?:what|how|why|when|where|who|which|can you|could you|explain|describe|list|tell me|give me)",
    re.IGNORECASE,
)


# Pronouns and bare references that signal a dangling reference needing context
_DANGLING_RE = re.compile(
    r"\b(one|ones|it|its|them|they|their|this|that|these|those)\b",
    re.IGNORECASE,
)

# Superlative/comparative adjectives that often precede a dangling "one"
_BARE_SUPERLATIVE_RE = re.compile(
    r"\bthe\s+(biggest|smallest|largest|fastest|slowest|best|worst|most|least|"
    r"longest|shortest|tallest|heaviest|lightest|oldest|newest|cheapest|"
    r"most expensive|closest|farthest|nearest)\b",
    re.IGNORECASE,
)


def _extract_topic(first_part: str, full_query: str) -> str:
    """Extract the topic noun phrase from the first sub-question.

    Args:
        first_part: The first sub-question text.
        full_query: The original unsplit query for fallback.

    Returns:
        A topic string like 'planets in the solar system', or empty
        string if no topic can be extracted.
    """
    text = first_part.lower().strip()
    # Strip common question prefixes
    prefixes = [
        r"^how many\s+", r"^what (?:is|are|was|were) (?:the |a |an )?\s*",
        r"^what\s+", r"^who (?:is|are|was|were)\s+",
        r"^where (?:is|are|was|were)\s+", r"^when (?:is|are|was|were)\s+",
        r"^can you (?:name|list|tell me about|describe|explain)\s+",
        r"^(?:name|list|tell me about|describe|explain)\s+",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Strip verb phrases that aren't part of the noun phrase
    text = re.sub(
        r"\s+(?:are|is|was|were|do|does|did|have|has|had)\s+"
        r"(?:there|here|it|they|we|you)\s*",
        " ", text, flags=re.IGNORECASE,
    )
    # Strip trailing verbs/prepositions
    text = re.sub(r"\s+(?:do|does|did|have|has|had|is|are|was|were)$", "", text)
    text = text.strip().rstrip("?.,;:")

    if len(text) > 2:
        return text
    return ""


def _carry_context(parts: List[str], full_query: str) -> List[str]:
    """Inject topic context into sub-questions with dangling references.

    When a sub-question contains pronouns like 'one', 'it', 'them' or
    bare superlatives like 'the biggest' without a noun, the topic from
    the first sub-question is appended to provide context.

    Args:
        parts: List of sub-question strings from the splitter.
        full_query: The original unsplit query.

    Returns:
        List of sub-questions with context injected where needed.
    """
    if len(parts) < 2:
        return parts

    topic = _extract_topic(parts[0], full_query)
    if not topic:
        return parts

    result = [parts[0]]
    for part in parts[1:]:
        has_dangling = _DANGLING_RE.search(part)
        has_bare_superlative = _BARE_SUPERLATIVE_RE.search(part)

        if has_dangling or has_bare_superlative:
            # Replace dangling "one"/"ones" with the topic noun
            enriched = re.sub(
                r"\b(the\s+(?:biggest|smallest|largest|fastest|slowest|best|worst|"
                r"most|least|longest|shortest|tallest|heaviest|lightest|oldest|"
                r"newest|cheapest|most expensive|closest|farthest|nearest))\s+"
                r"(one|ones?)\b",
                rf"\1 {topic}",
                part,
                flags=re.IGNORECASE,
            )
            # If no superlative+one pattern, just append topic as context
            if enriched == part:
                enriched = f"{part} (regarding {topic})"
            result.append(enriched)
        else:
            result.append(part)

    return result


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
        return _carry_context(topic_parts, query)

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
