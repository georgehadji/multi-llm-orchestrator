"""
LLM Re-ranking — Quality-based Result Re-ranking
=================================================

Implements LLM-based re-ranking for search results.
Based on QMD re-ranking architecture.

Uses the Orchestrator's existing LLM clients to score
search results for relevance.

Usage:
    from orchestrator.reranker import LLMReranker

    reranker = LLMReranker()

    # Re-rank search results
    results = [
        {"doc_id": "1", "content": "Python tutorial"},
        {"doc_id": "2", "content": "JavaScript guide"},
    ]

    ranked = await reranker.rerank(
        query="python programming",
        results=results,
        top_k=5,
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Re-ranking result for a single document."""
    doc_id: str
    original_rank: int
    new_rank: int
    relevance_score: float  # 0.0 to 1.0
    confidence: float  # Model confidence
    reasoning: str | None = None


class LLMReranker:
    """
    LLM-based re-ranking for search results.

    Uses the Orchestrator's LLM clients to score
    document relevance to a query.
    """

    # Re-ranking prompt template
    RERANK_PROMPT = """You are evaluating search result relevance.

Query: {query}

Document:
{content}

Rate relevance on scale 0-10:
- 10: Perfect match, directly answers query
- 7-9: Highly relevant, closely related
- 4-6: Moderately relevant, somewhat related
- 1-3: Low relevance, tangentially related
- 0: Irrelevant, not related at all

Respond with ONLY a JSON object:
{{
    "score": <0-10>,
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}
"""

    def __init__(
        self,
        model: str | None = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize re-ranker.

        Args:
            model: LLM model to use (default: uses orchestrator default)
            max_concurrent: Maximum concurrent re-ranking requests
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[RerankResult]:
        """
        Re-rank search results by relevance.

        Args:
            query: Search query
            results: List of search results to re-rank
            top_k: Maximum results to return
            min_score: Minimum relevance score threshold

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        # Score all results
        tasks = [
            self._score_result(query, result, i)
            for i, result in enumerate(results)
        ]

        # Run with concurrency limit
        scored = await asyncio.gather(*tasks)

        # Filter by min_score
        filtered = [s for s in scored if s.relevance_score >= min_score]

        # Sort by relevance score
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)

        # Take top_k
        top_results = filtered[:top_k]

        # Update ranks
        for i, result in enumerate(top_results):
            result.new_rank = i + 1

        logger.debug(f"Re-ranked {len(results)} results, returning {len(top_results)}")

        return top_results

    async def _score_result(
        self,
        query: str,
        result: dict[str, Any],
        original_rank: int,
    ) -> RerankResult:
        """Score a single result for relevance."""
        async with self._semaphore:
            try:
                # Get content to score
                content = result.get("content", "")
                if len(content) > 2000:
                    content = content[:2000] + "..."  # Truncate for efficiency

                # Build prompt
                prompt = self.RERANK_PROMPT.format(
                    query=query,
                    content=content,
                )

                # Call LLM
                score_data = await self._call_llm(prompt)

                # Parse response
                score = score_data.get("score", 5) / 10.0  # Normalize to 0-1
                confidence = score_data.get("confidence", 0.5)
                reasoning = score_data.get("reasoning")

                return RerankResult(
                    doc_id=result.get("doc_id", f"doc_{original_rank}"),
                    original_rank=original_rank + 1,
                    new_rank=original_rank + 1,  # Will be updated after sorting
                    relevance_score=score,
                    confidence=confidence,
                    reasoning=reasoning,
                )

            except Exception as e:
                logger.warning(f"Re-ranking failed for result {original_rank}: {e}")
                # Return default score on error
                return RerankResult(
                    doc_id=result.get("doc_id", f"doc_{original_rank}"),
                    original_rank=original_rank + 1,
                    new_rank=original_rank + 1,
                    relevance_score=0.5,  # Default middle score
                    confidence=0.3,
                    reasoning=f"Error: {str(e)}",
                )

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM for re-ranking score."""
        # Try to use orchestrator's client
        try:
            from .api_clients import UnifiedClient
            from .models import Model

            client = UnifiedClient()

            # Use a fast model for re-ranking
            model = self.model
            if not model:
                model = "gpt-4o-mini"  # Fast and cost-effective

            response = await client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a relevance scoring assistant. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200,
            )

            # Parse JSON response
            import json
            content = response.choices[0].message.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            return json.loads(content)

        except ImportError:
            logger.warning("UnifiedClient not available, using mock re-ranking")
            return self._mock_score(prompt)
        except Exception as e:
            logger.warning(f"LLM re-ranking error: {e}")
            return self._mock_score(prompt)

    def _mock_score(self, prompt: str) -> dict[str, Any]:
        """Mock scoring when LLM not available."""
        # Simple keyword-based scoring as fallback
        import re

        # Extract query terms
        query_match = re.search(r"Query: (.+)", prompt)
        content_match = re.search(r"Document:\n(.+)", prompt, re.DOTALL)

        if query_match and content_match:
            query = query_match.group(1).lower()
            content = content_match.group(1).lower()

            # Count query terms in content
            query_terms = query.split()
            matches = sum(1 for term in query_terms if term in content)
            score = min(1.0, matches / max(len(query_terms), 1))

            return {
                "score": int(score * 10),
                "confidence": 0.5,
                "reasoning": f"Keyword match: {matches}/{len(query_terms)} terms",
            }

        return {
            "score": 5,
            "confidence": 0.3,
            "reasoning": "Default score",
        }

    async def rerank_batch(
        self,
        queries: list[str],
        results_list: list[list[dict[str, Any]]],
        top_k: int = 10,
    ) -> list[list[RerankResult]]:
        """
        Re-rank multiple query results in batch.

        Args:
            queries: List of queries
            results_list: List of result lists (one per query)
            top_k: Maximum results per query

        Returns:
            List of re-ranked result lists
        """
        tasks = [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_list, strict=False)
        ]

        return await asyncio.gather(*tasks)


# Global re-ranker instance
_default_reranker: LLMReranker | None = None


def get_reranker(model: str | None = None) -> LLMReranker:
    """Get or create re-ranker instance."""
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = LLMReranker(model)
    return _default_reranker


async def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
    min_score: float = 0.3,
) -> list[RerankResult]:
    """Convenience function for re-ranking."""
    reranker = get_reranker()
    return await reranker.rerank(query, results, top_k, min_score)
