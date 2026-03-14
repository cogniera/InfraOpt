"""TemplateCache entry point. Provides the query() method."""

import asyncio
import logging
from typing import Dict, List

from templatecache.modules.cache_store import CacheStore
from templatecache.modules.cluster_router import ClusterRouter
from templatecache.modules.router import IntentRouter
from templatecache.modules.slot_engine import SlotEngine
from templatecache.utils.extractor import (
    detect_query_gaps,
    determine_variant,
    extract_template,
    split_multi_query,
)
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
        self._cluster_router = ClusterRouter()
        self._slot_engine = SlotEngine(self._cache_store)
        self._seeded = False

    async def _ensure_seeded(self) -> None:
        """Seed centroids on first startup if none exist in Redis.

        Builds cluster router if enough centroids exist. Blocks until
        seeding is complete. Do not serve queries until seeding is finished.

        Side effects:
            Writes seed data to Redis if no centroids exist.
            Builds cluster index from all centroids.
        """
        if self._seeded:
            return
        centroids = self._cache_store.get_all_intent_centroids()
        if not centroids:
            logger.info("No centroids found. Seeding defaults...")
            await self._router.seed_centroids(_DEFAULT_SEED_EXAMPLES)
            centroids = self._cache_store.get_all_intent_centroids()

        # Build cluster router from all centroids
        if centroids and not self._cluster_router.is_built:
            self._cluster_router.build(centroids)
            if self._cluster_router.is_built:
                logger.info(
                    "Cluster router active: %d clusters",
                    self._cluster_router.cluster_count,
                )
        self._seeded = True


    async def _query_single(self, prompt: str) -> Dict:
        """Process a single-topic query through the pipeline.

        Args:
            prompt: A single-topic user query.

        Returns:
            Dict with the full response contract.

        Side effects:
            May make LLM and embedding API calls. May write to Redis.
        """
        # Route query — use cluster router if built, else flat scan
        cluster_label = None
        if self._cluster_router.is_built:
            intent_id, variant, cluster_label = self._cluster_router.route(prompt)
        else:
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
                    logger.warning(
                        "Slot fill failed, serving cached response: %s",
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
                    estimated_full = len(response.split()) * 2
                    actual_used = from_inference * 40
                    savings = max(0.0, 1.0 - (actual_used / max(estimated_full, 1)))

                    if cluster_label:
                        stitch_info["cluster"] = cluster_label

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

    async def query(self, prompt: str) -> Dict:
        """Process a query through the TemplateCache pipeline.

        Splits multi-topic queries into sub-questions, routes each
        independently (cache hit or LLM miss), and combines results.
        Always returns a dict with the full response contract.

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

            sub_queries = split_multi_query(prompt)

            # Single query — use standard path
            if len(sub_queries) <= 1:
                return await self._query_single(prompt)

            # Multi-query — route each sub-question independently
            logger.info(
                "Multi-query detected: %d sub-questions from '%s'",
                len(sub_queries), prompt,
            )
            sub_results = []
            for sq in sub_queries:
                result = await self._query_single(sq)
                sub_results.append(result)

            # Combine results
            combined_parts = []
            total_from_cache = 0
            total_from_inference = 0
            total_estimated = 0
            total_actual = 0
            any_hit = False
            intent_ids = []
            sub_stitch = []

            for sq, result in zip(sub_queries, sub_results):
                combined_parts.append(f"**{sq.strip().capitalize()}:** {result['response']}")
                total_from_cache += result["slots_from_cache"]
                total_from_inference += result["slots_from_inference"]
                total_estimated += result["estimated_full_tokens"]
                total_actual += result["actual_tokens_used"]
                if result["cache_hit"]:
                    any_hit = True
                intent_ids.append(result.get("intent_id"))
                sub_stitch.append({
                    "sub_query": sq,
                    "cache_hit": result["cache_hit"],
                    "intent_id": result.get("intent_id"),
                })

            combined_response = "\n\n".join(combined_parts)
            savings = max(0.0, 1.0 - (total_actual / max(total_estimated, 1)))

            return {
                "response": combined_response,
                "cache_hit": any_hit,
                "intent_id": ", ".join(str(i) for i in intent_ids if i),
                "slots_from_cache": total_from_cache,
                "slots_from_inference": total_from_inference,
                "estimated_full_tokens": total_estimated,
                "actual_tokens_used": total_actual,
                "savings_ratio": savings,
                "stitch": {
                    "skeleton": combined_response,
                    "slot_fills": {},
                    "slot_sources": {},
                    "has_slots": False,
                    "multi_query": True,
                    "sub_results": sub_stitch,
                },
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