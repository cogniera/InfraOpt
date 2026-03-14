"""TemplateCache entry point. Provides the query() method."""

import asyncio
import logging
from typing import Dict, List

from templatecache.modules.cache_store import CacheStore
from templatecache.modules.router import IntentRouter
from templatecache.modules.slot_engine import SlotEngine
from templatecache.utils.extractor import detect_query_gaps, determine_variant, extract_template
from templatecache.utils.llm import llm_call

logger = logging.getLogger(__name__)

# Default seed examples covering 3 intent types and all 3 variants
_DEFAULT_SEED_EXAMPLES: List[Dict[str, str]] = [
    {
        "intent_id": "greeting",
        "query": "Hi there",
        "response": "Hello! How can I help you today?",
    },
    {
        "intent_id": "greeting",
        "query": "Hey",
        "response": "Hi! What can I do for you?",
    },
    {
        "intent_id": "explanation",
        "query": "Explain how photosynthesis works in detail",
        "response": (
            "Photosynthesis is the process by which plants convert sunlight into energy. "
            "During the light-dependent reactions, chlorophyll absorbs sunlight and splits "
            "water molecules, producing oxygen and ATP. In the Calvin cycle, CO2 is fixed "
            "into glucose using the ATP generated earlier."
        ),
    },
    {
        "intent_id": "explanation",
        "query": "Describe how neural networks learn",
        "response": (
            "Neural networks learn through a process called backpropagation. During forward "
            "pass, input data flows through layers of neurons. The loss function measures "
            "error, and gradients are computed backwards through the network to update weights."
        ),
    },
    {
        "intent_id": "listing",
        "query": "List the main programming paradigms",
        "response": (
            "1. Imperative programming\n"
            "2. Object-oriented programming\n"
            "3. Functional programming\n"
            "4. Declarative programming\n"
            "5. Logic programming"
        ),
    },
    {
        "intent_id": "listing",
        "query": "What are the planets in our solar system",
        "response": (
            "1. Mercury\n2. Venus\n3. Earth\n4. Mars\n"
            "5. Jupiter\n6. Saturn\n7. Uranus\n8. Neptune"
        ),
    },
]


class TemplateCache:
    """Main entry point for the TemplateCache system.

    Combines IntentRouter, SlotEngine, and CacheStore into a single
    drop-in caching primitive.
    """

    def __init__(self) -> None:
        """Initialize TemplateCache with all sub-modules."""
        self._cache_store = CacheStore()
        self._router = IntentRouter(self._cache_store)
        self._slot_engine = SlotEngine(self._cache_store)
        self._seeded = False

    async def _ensure_seeded(self) -> None:
        """Seed centroids on first startup if none exist in Redis.

        Blocks until seeding is complete. Do not serve queries until
        seeding is finished.

        Side effects:
            Writes seed data to Redis if no centroids exist.
        """
        if self._seeded:
            return
        centroids = self._cache_store.get_all_intent_centroids()
        if not centroids:
            logger.info("No centroids found. Seeding defaults...")
            await self._router.seed_centroids(_DEFAULT_SEED_EXAMPLES)
        self._seeded = True


    async def query(self, prompt: str) -> Dict:
        """Process a query through the TemplateCache pipeline.

        Always returns a dict with the full response contract. If any stage
        fails, catches the exception, logs it, and falls back to full
        generation with cache_hit: False.

        Args:
            prompt: The user query text.

        Returns:
            Dict with keys: response, cache_hit, intent_id, slots_from_cache,
            slots_from_inference, estimated_full_tokens, actual_tokens_used,
            savings_ratio.

        Side effects:
            May make LLM and embedding API calls. May write to Redis.
        """
        try:
            await self._ensure_seeded()

            # Route query
            intent_id, variant = self._router.route(prompt)

            if intent_id is not None:
                # Cache hit path
                template = self._cache_store.get_template(intent_id)
                if template is not None:
                    template.hit_count += 1
                    asyncio.create_task(
                        self._cache_store.write_back(template=template)
                    )

                    # Detect query aspects not covered by the cached response
                    gaps = detect_query_gaps(prompt, template.skeleton)

                    try:
                        response, from_cache, from_inference, stitch_info = (
                            await self._slot_engine.fill(
                                template, prompt, gaps=gaps
                            )
                        )
                    except Exception as slot_err:
                        # If supplement LLM call fails, serve cached part
                        logger.warning(
                            "Supplement slot fill failed, serving cached response: %s",
                            slot_err,
                        )
                        response = template.skeleton
                        from_cache, from_inference = 0, 0
                        stitch_info = {
                            "skeleton": template.skeleton,
                            "slot_fills": {},
                            "slot_sources": {},
                            "has_slots": False,
                            "gaps_detected": gaps,
                            "supplement_error": str(slot_err),
                        }

                    if response is not None:
                        # Estimate tokens
                        estimated_full = len(response.split()) * 2
                        actual_used = from_inference * 40  # rough estimate per slot
                        savings = 1.0 - (actual_used / max(estimated_full, 1))
                        savings = max(0.0, savings)

                        return {
                            "response": response,
                            "cache_hit": True,
                            "intent_id": intent_id,
                            "slots_from_cache": from_cache,
                            "slots_from_inference": from_inference,
                            "estimated_full_tokens": estimated_full,
                            "actual_tokens_used": actual_used,
                            "savings_ratio": savings,
                            "stitch": stitch_info,
                        }

            # Cache miss — full LLM generation
            miss_prompt = f"Answer concisely and directly.\n\n{prompt}"
            response_text = await llm_call(miss_prompt, max_tokens=512)

            # Extract template for future use
            skeleton, slots, dep_graph = extract_template(response_text)
            variant = determine_variant(prompt)

            if intent_id is None:
                import hashlib

                intent_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]

            from templatecache.models.template import ResponseTemplate
            from templatecache.models.intent import IntentCentroid
            from templatecache.utils.embedder import embed

            new_template = ResponseTemplate(
                intent_id=intent_id,
                skeleton=skeleton,
                slots=slots,
                dependency_graph=dep_graph,
                variant=variant,
            )

            query_emb = embed(prompt)
            new_centroid = IntentCentroid(
                intent_id=intent_id,
                centroid_embedding=query_emb,
                template_id=intent_id,
                variant=variant,
                query_count=1,
            )

            # Async write-back — never block the response path
            asyncio.create_task(
                self._cache_store.write_back(
                    template=new_template, centroid=new_centroid
                )
            )

            estimated_full = len(response_text.split()) * 2
            return {
                "response": response_text,
                "cache_hit": False,
                "intent_id": intent_id,
                "slots_from_cache": 0,
                "slots_from_inference": 0,
                "estimated_full_tokens": estimated_full,
                "actual_tokens_used": estimated_full,
                "savings_ratio": 0.0,
            }

        except Exception as e:
            logger.exception("Pipeline error, falling back to full generation: %s", e)
            try:
                fallback_prompt = f"Answer concisely and directly.\n\n{prompt}"
                response_text = await llm_call(fallback_prompt, max_tokens=512)
            except Exception:
                response_text = "An error occurred processing your request."

            estimated_full = len(response_text.split()) * 2
            return {
                "response": response_text,
                "cache_hit": False,
                "intent_id": None,
                "slots_from_cache": 0,
                "slots_from_inference": 0,
                "estimated_full_tokens": estimated_full,
                "actual_tokens_used": estimated_full,
                "savings_ratio": 0.0,
            }