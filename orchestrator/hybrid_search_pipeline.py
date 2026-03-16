"""
HybridSearchPipeline — BM25 + Vector + RRF + Query Expansion
=============================================================
Unified search pipeline that:
  1. Expands the query into multiple phrasings (via QueryExpander)
  2. Runs BM25 and vector search for each phrasing in parallel
  3. Fuses all result lists with Reciprocal Rank Fusion (k=60)
  4. Optionally reranks the fused results via LLMReranker
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .bm25_search import BM25Search, SearchResult
from .log_config import get_logger

logger = get_logger(__name__)

_RRF_K = 60


class HybridSearchPipeline:
    """
    Full hybrid search pipeline: QueryExpander → BM25 + Vector → RRF → Reranker.

    All dependencies are optional; the pipeline degrades gracefully:
    - No query_expander → use original query only
    - No knowledge_base → skip vector search (BM25 only)
    - No reranker → skip reranking step
    """

    def __init__(
        self,
        bm25_search: BM25Search,
        knowledge_base: Optional[Any] = None,
        reranker: Optional[Any] = None,
        query_expander: Optional[Any] = None,
    ) -> None:
        self._bm25 = bm25_search
        self._kb = knowledge_base
        self._reranker = reranker
        self._expander = query_expander

    # ── Public API ──────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        top_k: int = 10,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
    ) -> List[SearchResult]:
        """
        Execute the full hybrid search pipeline.

        Args:
            query: Original search query.
            project_id: Optional filter for BM25Search.
            top_k: Maximum results to return.
            use_query_expansion: Whether to expand query via LLM.
            use_reranking: Whether to apply LLM reranking after RRF.

        Returns:
            List of SearchResult ordered by relevance (best first).
        """
        # 1. Query expansion
        if use_query_expansion and self._expander is not None:
            queries = await self._expander.expand(query)
        else:
            queries = [query]

        # 2. Gather result lists (BM25 + vector) for each query variant
        fetch_limit = top_k * 3  # oversample before fusion
        all_result_lists: List[List[SearchResult]] = []

        gather_tasks = []
        for q in queries:
            gather_tasks.append(self._fetch_bm25(q, project_id, fetch_limit))
            if self._kb is not None:
                gather_tasks.append(self._fetch_vector(q, fetch_limit))

        fetched = await asyncio.gather(*gather_tasks, return_exceptions=True)
        for item in fetched:
            if isinstance(item, Exception):
                logger.warning("search fetch failed: %s", item)
            elif item:
                all_result_lists.append(item)

        if not all_result_lists:
            return []

        # 3. RRF fusion
        fused = self._rrf_merge(all_result_lists)

        # 4. Optional reranking
        if use_reranking and self._reranker is not None and fused:
            dicts = [
                {"doc_id": r.doc_id, "content": r.content, "title": r.title}
                for r in fused[: top_k * 2]
            ]
            try:
                reranked = await self._reranker.rerank(query, dicts, top_k=top_k)
                # Map reranked results back to SearchResult by doc_id
                fused_map = {r.doc_id: r for r in fused}
                result = []
                for rr in reranked:
                    sr = fused_map.get(rr.doc_id)
                    if sr:
                        # BUG-003 FIX: create a new SearchResult instead of mutating
                        # sr in-place.  If reranking raises mid-loop, fused[:top_k]
                        # in the except branch still contains unmodified RRF results.
                        result.append(SearchResult(
                            doc_id=sr.doc_id,
                            project_id=sr.project_id,
                            title=sr.title,
                            content=sr.content,
                            score=rr.relevance_score,
                            rank=rr.new_rank,
                            metadata=sr.metadata,
                        ))
                return result[:top_k]
            except Exception as exc:
                logger.warning("reranking failed (%s) — returning RRF results", exc)

        return fused[:top_k]

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _fetch_bm25(
        self, query: str, project_id: Optional[str], limit: int
    ) -> List[SearchResult]:
        """Run BM25 search and return SearchResult list."""
        return await self._bm25.bm25_search(query, project_id=project_id, limit=limit)

    async def _fetch_vector(self, query: str, limit: int) -> List[SearchResult]:
        """Run vector search via KnowledgeBase and convert to SearchResult."""
        artifacts = await self._kb.find_similar(query, top_k=limit)
        results = []
        for i, artifact in enumerate(artifacts):
            results.append(SearchResult(
                doc_id=artifact.id,
                project_id=getattr(artifact, "source_project", "") or "",
                title=getattr(artifact, "title", "") or "",
                content=artifact.content,
                score=getattr(artifact, "similarity_score", 0.5),
                rank=i + 1,
                metadata=getattr(artifact, "context", {}) or {},
            ))
        return results

    def _rrf_merge(self, result_lists: List[List[SearchResult]]) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion across multiple result lists.

        Score formula: Σ(1 / (k + rank)) for each list the doc appears in.
        """
        scores: dict[str, tuple[float, SearchResult]] = {}
        for result_list in result_lists:
            for result in result_list:
                rrf = 1.0 / (_RRF_K + result.rank)
                if result.doc_id in scores:
                    prev_score, prev_doc = scores[result.doc_id]
                    scores[result.doc_id] = (prev_score + rrf, prev_doc)
                else:
                    scores[result.doc_id] = (rrf, result)

        combined = [
            SearchResult(
                doc_id=doc_id,
                project_id=doc.project_id,
                title=doc.title,
                content=doc.content,
                score=score,
                rank=0,
                metadata=doc.metadata,
            )
            for doc_id, (score, doc) in scores.items()
        ]
        combined.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(combined):
            r.rank = i + 1
        return combined
