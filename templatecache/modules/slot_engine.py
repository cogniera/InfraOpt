"""SlotEngine — fills template slots with query-specific content."""

import asyncio
import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

import random

from templatecache.config import (
    SLOT_BLEND_ENABLED,
    SLOT_BLEND_THRESHOLD,
    SLOT_CONFIDENCE_THRESHOLD,
    SLOT_TRANSFER_ENABLED,
    SLOT_TRANSFER_PENALTY,
    UNCERTAIN_SLOT_FALLBACK_RATIO,
)
from templatecache.models.slot import SlotRecord
from templatecache.models.template import ResponseTemplate
from templatecache.modules.cache_store import CacheStore
from templatecache.utils.embedder import cosine_similarity, embed
from templatecache.utils.extractor import classify_slot
from templatecache.utils.llm import llm_call

logger = logging.getLogger(__name__)

# Known slot type suffixes, ordered longest-first for matching
_KNOWN_TYPES = [
    "quoted_content", "currency", "number", "date", "boilerplate",
]


def _extract_slot_type(slot_name: str) -> str:
    """Extract the slot type from a semantic slot name.

    Checks if the slot name ends with a known type suffix.
    E.g. 'creator_quoted_content' → 'quoted_content',
    'limit_currency' → 'currency', 'year_founded_number' → 'number'.
    Falls back to 'boilerplate' if no known suffix matches.

    Args:
        slot_name: The full slot name.

    Returns:
        The slot type string.
    """
    for t in _KNOWN_TYPES:
        if slot_name.endswith(f"_{t}") or slot_name == t:
            return t
    # Legacy format: number_0, currency_1 — check if prefix is a type
    prefix = slot_name.rsplit("_", 1)[0] if "_" in slot_name else slot_name
    if prefix in _KNOWN_TYPES:
        return prefix
    return "boilerplate"


class SlotEngine:
    """Fills template slots with query-specific content.

    Loads the template for an intent, checks slot confidence, fills uncertain
    slots via targeted LLM calls in dependency order, stitches fills into the
    skeleton.
    """

    def __init__(self, cache_store: CacheStore) -> None:
        """Initialize SlotEngine.

        Args:
            cache_store: CacheStore instance for Redis access.
        """
        self._cache_store = cache_store

    def _context_hash(self, query: str) -> str:
        """Compute a stable hash from the query for slot lookups.

        Args:
            query: The user query text.

        Returns:
            SHA-256 hex digest of the query.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _dependency_order(self, template: ResponseTemplate) -> List[str]:
        """Return slots in dependency order (dependencies first).

        Args:
            template: The ResponseTemplate with slot and dependency info.

        Returns:
            Ordered list of slot names safe for sequential filling.
        """
        ordered: List[str] = []
        visited: set = set()

        def visit(slot: str) -> None:
            if slot in visited:
                return
            visited.add(slot)
            for dep in template.dependency_graph.get(slot, []):
                visit(dep)
            ordered.append(slot)

        for slot in template.slots:
            visit(slot)
        return ordered

    def _build_slot_prompt(
        self, query: str, slot_name: str, filled: Dict[str, str], template: ResponseTemplate
    ) -> str:
        """Build a targeted prompt for filling a single slot.

        Args:
            query: The user query.
            slot_name: Name of the slot to fill.
            filled: Already-filled slot values (dependencies).
            template: The template being filled.

        Returns:
            A prompt string for the LLM.
        """
        deps = ""
        if filled:
            deps = " Context: " + "; ".join(f"{k}={v}" for k, v in filled.items()) + "."
        return (
            f"Q: {query}{deps}\n"
            f"Fill the slot '{slot_name}'. Reply with ONLY the value, "
            f"no brackets, no quotes, nothing else."
        )


    @staticmethod
    def _clean_fill_value(value: str) -> str:
        """Clean an LLM-generated slot fill value.

        Strips whitespace, removes surrounding brackets (LLMs sometimes
        mimic the [slot_name] format), and removes surrounding quotes.

        Args:
            value: Raw fill value from the LLM.

        Returns:
            Cleaned fill value.
        """
        v = value.strip()
        # Remove surrounding brackets: [599] → 599
        if v.startswith("[") and v.endswith("]"):
            v = v[1:-1]
        # Remove surrounding quotes: "hello" → hello
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        return v.strip()


    def _transfer_slot(
        self,
        slot_name: str,
        slot_type: str,
        query_embedding: List[float],
    ) -> Tuple[Optional[str], float]:
        """Attempt to fill a slot by transferring a value from another template.

        Searches all SlotRecords of the same type across all templates.
        Computes similarity between each candidate's fill_embedding and the
        current query_embedding. Applies SLOT_TRANSFER_PENALTY to the score.
        Returns the best fill value if it exceeds SLOT_CONFIDENCE_THRESHOLD
        after penalty.

        Args:
            slot_name: The slot name to fill.
            slot_type: The classified type of the slot (e.g. 'numeric').
            query_embedding: Embedding vector of the current query.

        Returns:
            Tuple of (fill_value, penalised_score) if a candidate qualifies,
            or (None, 0.0) if no candidate qualifies or transfer is disabled.
        """
        if not SLOT_TRANSFER_ENABLED:
            return None, 0.0

        candidates = self._cache_store.get_slots_by_type(slot_type)
        if not candidates:
            return None, 0.0

        best_value: Optional[str] = None
        best_score: float = 0.0

        for record in candidates:
            if not record.fill_embedding:
                continue
            raw_sim = cosine_similarity(query_embedding, record.fill_embedding)
            penalised = raw_sim - SLOT_TRANSFER_PENALTY
            if penalised > best_score and penalised >= SLOT_CONFIDENCE_THRESHOLD:
                best_score = penalised
                best_value = record.fill_value

        if best_value is not None:
            logger.info(
                "Transfer slot '%s' (type=%s): score=%.3f",
                slot_name, slot_type, best_score,
            )

        return best_value, best_score


    async def _blend_fills(
        self,
        slot_name: str,
        cached_value: str,
        fresh_value: str,
        confidence: float,
    ) -> Tuple[str, Dict]:
        """Select between a cached and fresh fill using confidence weighting.

        Uses the confidence score as the probability of selecting the cached
        value. Higher confidence → more likely to keep the cached value.

        Args:
            slot_name: The slot name being filled.
            cached_value: The cached fill value.
            fresh_value: The freshly LLM-generated fill value.
            confidence: The confidence score (0.0 to 1.0).

        Returns:
            Tuple of (selected_value, blend_record) where blend_record
            contains cached_value, fresh_value, confidence, and which
            was selected.
        """
        use_cached = random.random() < confidence
        selected = cached_value if use_cached else fresh_value
        blend_record = {
            "cached_value": cached_value,
            "fresh_value": fresh_value,
            "confidence": confidence,
            "selected": "cached" if use_cached else "fresh",
        }
        logger.info(
            "Blend slot '%s': confidence=%.3f, selected=%s",
            slot_name, confidence, blend_record["selected"],
        )
        return selected, blend_record


    def _build_supplement_prompt(
        self, query: str, gap: str, existing_response: str
    ) -> str:
        """Build a prompt for generating content to fill a query gap.

        Args:
            query: The full user query.
            gap: The specific aspect of the query not covered.
            existing_response: The cached response already being served.

        Returns:
            A prompt string for the LLM.
        """
        # Truncate existing response to save input tokens
        truncated = existing_response[:300]
        return (
            f"Already answered: {truncated}\n"
            f"Now answer ONLY this part: {gap}\n"
            f"Be concise. Do not repeat anything above."
        )

    def _stitch(self, skeleton: str, fills: Dict[str, str]) -> str:
        """Replace all [slot_name] markers in the skeleton with fill values.

        Args:
            skeleton: Template skeleton with [slot_name] placeholders.
            fills: Mapping from slot name to fill value.

        Returns:
            Final response string with all markers replaced. Any remaining
            unreplaced markers are removed to prevent leaking internals.
        """
        result = skeleton
        for slot_name, value in fills.items():
            result = result.replace(f"[{slot_name}]", value)
        # Safety: remove any remaining [slot_name] markers that weren't
        # filled (e.g. slot name mismatch, missing fill)
        result = re.sub(r"\[[a-z][a-z0-9_]*\]", "", result)
        return result

    async def fill(
        self, template: ResponseTemplate, query: str,
        gaps: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], int, int, Dict]:
        """Fill a template's slots for the given query.

        Follows slot filling rules: dependency order, confidence check,
        fallback ratio check, sequential filling. If gaps are provided,
        supplement slots are added and filled via LLM to cover query
        aspects not addressed by the cached template.

        Args:
            template: The ResponseTemplate to fill.
            query: The user query providing context.
            gaps: Optional list of query aspects not covered by the template.

        Returns:
            Tuple of (response, slots_from_cache, slots_from_inference, stitch_info).
            stitch_info contains skeleton, slot_fills, and slot_sources for visualization.
            Returns (None, 0, 0, {}) if fallback to full generation is needed.

        Side effects:
            May make LLM calls for uncertain slots. May write back slot
            records to CacheStore.
        """
        ctx_hash = self._context_hash(query)
        ordered_slots = self._dependency_order(template)
        has_gaps = gaps is not None and len(gaps) > 0

        # If template has no slots and no gaps, return the skeleton directly
        if not ordered_slots and not has_gaps:
            stitch_info = {
                "skeleton": template.skeleton,
                "slot_fills": {},
                "slot_sources": {},
                "has_slots": False,
            }
            return template.skeleton, 0, 0, stitch_info

        # Phase 1: Three-state confidence classification
        #   score >= SLOT_BLEND_THRESHOLD       → confident, serve cached
        #   SLOT_CONFIDENCE_THRESHOLD <= score   → blend zone
        #   score < SLOT_CONFIDENCE_THRESHOLD    → uncertain
        uncertain_slots: List[str] = []
        blend_slots: Dict[str, SlotRecord] = {}  # slot_name → record
        cached_fills: Dict[str, str] = {}
        slot_sources: Dict[str, str] = {}

        for slot_name in ordered_slots:
            record = self._cache_store.get_slot_confidence(slot_name, ctx_hash)
            if record is not None and record.similarity_score >= SLOT_BLEND_THRESHOLD:
                cached_fills[slot_name] = record.fill_value
                slot_sources[slot_name] = "cache"
            elif (
                SLOT_BLEND_ENABLED
                and record is not None
                and record.similarity_score >= SLOT_CONFIDENCE_THRESHOLD
            ):
                blend_slots[slot_name] = record
            else:
                uncertain_slots.append(slot_name)

        # Phase 2: Check fallback ratio — abort template filling if too
        # many slots are uncertain, UNLESS all slots are uncertain (new
        # template, first hit). In that case, fill them all — they were
        # just extracted from this context.
        total = len(ordered_slots)
        all_uncertain = len(uncertain_slots) == total
        if (
            not all_uncertain
            and total >= 3
            and len(uncertain_slots) / total > UNCERTAIN_SLOT_FALLBACK_RATIO
        ):
            logger.info(
                "Uncertain ratio %.2f exceeds threshold, falling back to full generation",
                len(uncertain_slots) / total,
            )
            return None, 0, 0, {}

        # Phase 2.5: Process blend zone slots — generate fresh fill and
        # select between cached and fresh using confidence as probability.
        fills: Dict[str, str] = dict(cached_fills)
        slots_from_blend = 0
        blend_candidates: Dict[str, Dict] = {}

        for slot_name, record in blend_slots.items():
            prompt = self._build_slot_prompt(query, slot_name, fills, template)
            fresh_value = await llm_call(prompt, max_tokens=80)
            fresh_value = self._clean_fill_value(fresh_value)
            selected, blend_record = await self._blend_fills(
                slot_name, record.fill_value, fresh_value, record.similarity_score
            )
            fills[slot_name] = selected
            slot_sources[slot_name] = "blend"
            blend_candidates[slot_name] = blend_record
            slots_from_blend += 1

        # Phase 3: Fill uncertain slots sequentially in dependency order
        # Try cross-query transfer before falling back to LLM.
        slots_from_inference = 0
        slots_from_transfer = 0
        query_embedding = embed(query)

        for slot_name in ordered_slots:
            if slot_name in fills:
                continue

            # Determine slot type from the name suffix
            # Semantic names end with the type: e.g. creator_quoted_content,
            # limit_currency, year_founded_number
            slot_type = _extract_slot_type(slot_name)

            # Attempt cross-query transfer first
            transfer_value, transfer_score = self._transfer_slot(
                slot_name, slot_type, query_embedding
            )
            if transfer_value is not None:
                fills[slot_name] = transfer_value
                slot_sources[slot_name] = "transfer"
                slots_from_transfer += 1
                continue

            # Fall back to LLM
            prompt = self._build_slot_prompt(query, slot_name, fills, template)
            fill_value = await llm_call(prompt, max_tokens=80)
            fill_value = self._clean_fill_value(fill_value)
            fills[slot_name] = fill_value
            slot_sources[slot_name] = "inference"
            slots_from_inference += 1

            # Write back new slot record (non-blocking handled by caller)
            fill_emb = embed(fill_value.strip())
            new_record = SlotRecord(
                slot_id=slot_name,
                context_hash=ctx_hash,
                fill_value=fill_value.strip(),
                fill_embedding=fill_emb,
                similarity_score=1.0,
                slot_type=classify_slot(fill_value.strip()),
            )
            asyncio.create_task(self._cache_store.write_back(slot_record=new_record))

        # Phase 4: Stitch existing slots
        response = self._stitch(template.skeleton, fills)

        # Phase 5: Fill supplement slots for uncovered query gaps
        supplement_fills: Dict[str, str] = {}
        full_query_gap = False
        if has_gaps:
            # If the gap IS the entire query (single aspect, same text as
            # the query), the cached response is irrelevant — replace it
            # entirely with the LLM-generated supplement.
            full_query_gap = (
                len(gaps) == 1
                and gaps[0].lower().strip() == query.lower().strip()
            )

            for i, gap in enumerate(gaps):
                slot_name = f"supplement_{i}"
                if full_query_gap:
                    # Full replacement: generate a direct answer
                    prompt = (
                        f"Answer this question concisely:\n{gap}"
                    )
                else:
                    prompt = self._build_supplement_prompt(query, gap, response)
                fill_value = await llm_call(prompt, max_tokens=150)
                supplement_fills[slot_name] = fill_value.strip()
                slot_sources[slot_name] = "inference"
                slots_from_inference += 1

            if full_query_gap:
                # Replace cached response entirely with supplement
                response = supplement_fills["supplement_0"]
                fills = supplement_fills
            else:
                # Append supplements to cached response
                skeleton_with_supplements = response
                for slot_name, value in supplement_fills.items():
                    skeleton_with_supplements += f"\n\n[{slot_name}]"
                    fills[slot_name] = value
                response = self._stitch(skeleton_with_supplements, fills)

        if full_query_gap:
            skeleton_for_viz = "[supplement_0]"
        else:
            skeleton_for_viz = template.skeleton
            if has_gaps:
                for slot_name in supplement_fills:
                    skeleton_for_viz += f"\n\n[{slot_name}]"

        stitch_info = {
            "skeleton": skeleton_for_viz,
            "slot_fills": fills,
            "slot_sources": slot_sources,
            "has_slots": bool(fills),
            "gaps_detected": gaps or [],
            "full_query_gap": full_query_gap,
            "slots_from_transfer": slots_from_transfer,
            "slots_from_blend": slots_from_blend,
            "blend_candidates": blend_candidates,
        }
        return response, len(cached_fills), slots_from_inference, stitch_info
