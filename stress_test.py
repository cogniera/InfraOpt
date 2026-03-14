"""Stress test: run many queries through the pipeline and check for broken output."""

import asyncio
import json
import sys
import time
import httpx

QUERIES = [
    # Code examples
    "write hello world in python",
    "write hello world in rust",
    "write a class in java",
    "write fizzbuzz",
    "write a for loop in javascript",
    "write error handling in go",
    "write a Dockerfile for python",
    "write a unit test in python",
    # Disambiguation
    "who invented Python",
    "when was the internet created",
    "why is Python slow",
    "what is the best programming language",
    "pros and cons of microservices",
    "alternatives to Docker",
    "how long does it take to learn Python",
    "will AI replace programmers",
    "should I use React or Vue",
    # Concepts
    "what is a decorator in python",
    "what is a closure in javascript",
    "how does TCP work",
    "what is Docker",
    "what is a linked list",
    "difference between TCP and UDP",
    "difference between REST and GraphQL",
    # Comparative
    "what is the closest star to earth",
    "what is the tallest building",
    "what is the fastest animal",
    "what is the largest ocean",
    # Multi-query
    "what is Python and what is JavaScript",
    "explain TCP, UDP, and HTTP",
    "write hello world in python and explain what it does",
    # Edge cases
    "what is a pointer in C",
    "what is the speed of light",
    "how does DNS work",
    "what is a black hole",
    "what is inflation",
    # Rephrased (should hit after first pass)
    "write hello world in python",
    "who created Python",
    "what is the fastest animal on earth",
]

BROKEN_MARKERS = ["[number_", "[quoted_", "[currency_", "[name_", "[supplement_",
                   "[a_", "[b_", "[c_"]


async def run_stress_test():
    passed = 0
    failed = 0
    errors = []
    total_savings = 0.0
    cache_hits = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, query in enumerate(QUERIES):
            try:
                t0 = time.time()
                resp = await client.post(
                    "http://localhost:8000/query",
                    json={"prompt": query},
                )
                elapsed = time.time() - t0
                data = resp.json()

                response_text = data.get("response", "")
                savings = data.get("savings_ratio", 0)
                hit = data.get("cache_hit", False)

                # Check for broken slot markers
                has_broken = any(m in response_text for m in BROKEN_MARKERS)
                # Check for empty response
                is_empty = len(response_text.strip()) == 0
                # Check for bracket artifacts like $[899]
                import re
                bracket_artifacts = re.findall(r'.\[[a-z_]*\d+\]', response_text)

                issues = []
                if has_broken:
                    issues.append(f"BROKEN SLOT: {[m for m in BROKEN_MARKERS if m in response_text]}")
                if is_empty:
                    issues.append("EMPTY RESPONSE")
                if bracket_artifacts:
                    issues.append(f"BRACKET ARTIFACTS: {bracket_artifacts}")

                if issues:
                    failed += 1
                    icon = "❌"
                    errors.append((query, issues, response_text[:200]))
                else:
                    passed += 1
                    icon = "✅"

                hit_icon = "HIT" if hit else "MISS"
                total_savings += savings
                if hit:
                    cache_hits += 1

                print(f"  {icon} [{hit_icon:4s}] {elapsed:.1f}s {query[:50]}")
                if issues:
                    for issue in issues:
                        print(f"       ⚠️  {issue}")

            except Exception as e:
                failed += 1
                errors.append((query, [str(e)], ""))
                print(f"  💥 ERROR: {query[:50]} — {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(QUERIES)} total")
    print(f"Cache hits: {cache_hits}/{len(QUERIES)} ({cache_hits/len(QUERIES)*100:.0f}%)")
    print(f"Avg savings: {total_savings/len(QUERIES)*100:.1f}%")

    if errors:
        print(f"\n{'='*60}")
        print("FAILURES:")
        for query, issues, resp in errors:
            print(f"\n  Query: {query}")
            for issue in issues:
                print(f"    {issue}")
            if resp:
                print(f"    Response: {resp[:150]}")

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_stress_test())
    sys.exit(0 if ok else 1)

