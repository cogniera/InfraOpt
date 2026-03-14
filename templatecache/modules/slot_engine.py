"""SlotEngine — fills template slots with query-specific content."""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from templatecache.config import (
    SLOT_CONFIDENCE_THRESHOLD,
    UNCERTAIN_SLOT_FALLBACK_RATIO,
)
from templatecache.models.slot import SlotRecord
from templatecache.models.template import ResponseTemplate
from templatecache.modules.cache_store import CacheStore
from templatecache.utils.embedder import cosine_similarity, embed
from templatecache.utils.llm import llm_call

logger = logging.getLogger(__name__)


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
            f"Fill [{slot_name}]. Reply with ONLY the value, nothing else."
        )

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
            Final response string with all markers replaced.
        """
        result = skeleton
        for slot_name, value in fills.items():
            result = result.replace(f"[{slot_name}]", value)
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

        # Phase 1: Check confidence for all slots
        uncertain_slots: List[str] = []
        cached_fills: Dict[str, str] = {}
        slot_sources: Dict[str, str] = {}

        for slot_name in ordered_slots:
            record = self._cache_store.get_slot_confidence(slot_name, ctx_hash)
            if record is not None and record.similarity_score >= SLOT_CONFIDENCE_THRESHOLD:
                cached_fills[slot_name] = record.fill_value
                slot_sources[slot_name] = "cache"
            else:
                uncertain_slots.append(slot_name)

        # Phase 2: Check fallback ratio — only applies when there are
        # enough slots for the ratio to be meaningful (3+). Templates with
        # 1-2 slots should always attempt to fill them.
        total = len(ordered_slots)
        if total >= 3 and len(uncertain_slots) / total > UNCERTAIN_SLOT_FALLBACK_RATIO:
            logger.info(
                "Uncertain ratio %.2f exceeds threshold, falling back to full generation",
                len(uncertain_slots) / total,
            )
            return None, 0, 0, {}

        # Phase 3: Fill uncertain slots sequentially in dependency order
        fills: Dict[str, str] = dict(cached_fills)
        slots_from_inference = 0

        for slot_name in ordered_slots:
            if slot_name in fills:
                continue
            prompt = self._build_slot_prompt(query, slot_name, fills, template)
            fill_value = await llm_call(prompt, max_tokens=80)
            fills[slot_name] = fill_value.strip()
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
        }
        return response, len(cached_fills), slots_from_inference, stitch_info
