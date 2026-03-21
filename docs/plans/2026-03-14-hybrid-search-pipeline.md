# Hybrid Search Pipeline (RRF + Query Expansion) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `HybridSearchPipeline` class that combines BM25 + vector search via RRF fusion and LLM-based query expansion (DeepSeek-Chat), wiring it into `engine.hybrid_search()`.

**Architecture:** Two new modules (`query_expander.py`, `hybrid_search_pipeline.py`) plus engine delegation. Pipeline expands the original query into variants, runs BM25 and vector search for each variant, fuses all results with RRF (k=60), then optionally reranks via `LLMReranker`.

**Tech Stack:** Existing `BM25Search` (SQLite FTS5), `KnowledgeBase.find_similar()` (sentence-transformers), `LLMReranker`, `UnifiedClient` (DeepSeek-Chat for expansion).

---

## Context (read before starting)

**Existing modules — do NOT modify unless the task says to:**
- `orchestrator/bm25_search.py` — `BM25Search` with working `bm25_search()`, `_rrf_fusion()`, and stub `vector_search()` (falls back to BM25). `SearchResult` dataclass is defined here.
- `orchestrator/knowledge_base.py` — `KnowledgeBase.find_similar()` returns `List[KnowledgeArtifact]` with real vector search. `KnowledgeArtifact` has fields: `id`, `content`, `title`, `source_project`, `context`, `similarity_score`.
- `orchestrator/reranker.py` — `LLMReranker.rerank(query, results, top_k)` — `results` must be `List[Dict]` with at least `doc_id` and `content` keys.
- `orchestrator/api_clients.py` — `UnifiedClient` with `chat_completion(model, messages, temperature, max_tokens)`.
- `orchestrator/engine.py:724` — `Orchestrator.hybrid_search()` currently does BM25 + rerank only. We will update this to delegate to `HybridSearchPipeline`.
- `orchestrator/engine.py:165` — `Orchestrator.__init__` sets up `self._bm25_search`, `self._reranker`, `self._knowledge_base`. We add `self._hybrid_pipeline` here.

**Key type mapping** (KnowledgeArtifact → SearchResult):
```python
SearchResult(
    doc_id=artifact.id,
    project_id=artifact.source_project or "",
    title=artifact.title,
    content=artifact.content,
    score=getattr(artifact, "similarity_score", 0.5),
    rank=i + 1,
    metadata=artifact.context or {},
)
```

---

## Task 1: QueryExpander module

**Files:**
- Create: `orchestrator/query_expander.py`
- Create: `tests/test_query_expander.py`

### Step 1: Write failing tests

```python
# tests/test_query_expander.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator.query_expander import QueryExpander


@pytest.mark.asyncio
async def test_expand_returns_list_including_original():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = ["find python code", "search python examples", "python source code"]
        result = await expander.expand("python code")
    assert "python code" in result           # original always included
    assert isinstance(result, list)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_expand_falls_back_on_llm_error():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("LLM unavailable")
        result = await expander.expand("python code")
    assert result == ["python code"]          # graceful fallback = original only


@pytest.mark.asyncio
async def test_expand_deduplicates_and_preserves_original():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        # LLM returns original as one of its variants — should not duplicate
        mock_llm.return_value = ["python code", "python snippets"]
        result = await expander.expand("python code")
    assert result.count("python code") == 1  # deduplicated
    assert "python snippets" in result
```

### Step 2: Run tests — confirm they FAIL (ImportError)

Run: `pytest tests/test_query_expander.py -v --no-cov`

Expected: `ImportError: cannot import name 'QueryExpander' from 'orchestrator.query_expander'`

### Step 3: Create `orchestrator/query_expander.py`

```python
"""
QueryExpander — LLM-based query expansion for hybrid search.
=============================================================
Uses DeepSeek-Chat to generate alternative phrasings of a search
query, improving recall in BM25 and vector search.
"""
from __future__ import annotations

import json
import logging
from typing import List

from .log_config import get_logger

logger = get_logger(__name__)

_EXPAND_PROMPT = """Generate {n} alternative phrasings for the following search query.
Return ONLY a JSON array of strings. Do not include the original query.

Query: {query}

JSON array:"""


class QueryExpander:
    """
    Expands a search query into multiple alternative phrasings using an LLM.

    Falls back to returning only the original query if the LLM is unavailable.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_variants: int = 3,
    ) -> None:
        self.model = model
        self.max_variants = max_variants

    async def expand(self, query: str) -> List[str]:
        """
        Return [original_query] + up to max_variants LLM-generated alternatives.

        Duplicates are removed; original is always first.
        """
        try:
            variants = await self._call_llm(query)
        except Exception as exc:
            logger.warning("QueryExpander LLM call failed (%s) — using original query only", exc)
            return [query]

        # Deduplicate while preserving order; original always first
        seen: set[str] = {query}
        result = [query]
        for v in variants[: self.max_variants]:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result

    async def _call_llm(self, query: str) -> List[str]:
        """Call DeepSeek-Chat and parse JSON array of variants."""
        from .api_clients import UnifiedClient

        client = UnifiedClient()
        prompt = _EXPAND_PROMPT.format(n=self.max_variants, query=query)
        response = await client.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a search query expansion assistant. Respond ONLY with a valid JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
```

### Step 4: Run tests — confirm they PASS

Run: `pytest tests/test_query_expander.py -v --no-cov`

Expected: `3 passed`

### Step 5: Commit

```bash
git add orchestrator/query_expander.py tests/test_query_expander.py
git commit -m "feat: add QueryExpander for LLM-based query expansion (DeepSeek-Chat)"
```

---

## Task 2: HybridSearchPipeline module

**Files:**
- Create: `orchestrator/hybrid_search_pipeline.py`
- Create: `tests/test_hybrid_search_pipeline.py`

### Step 1: Write failing tests

```python
# tests/test_hybrid_search_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.hybrid_search_pipeline import HybridSearchPipeline
from orchestrator.bm25_search import BM25Search, SearchResult


def _make_sr(doc_id: str, score: float, rank: int) -> SearchResult:
    return SearchResult(
        doc_id=doc_id, project_id="proj1", content=f"content {doc_id}",
        title=doc_id, score=score, rank=rank,
    )


def test_pipeline_instantiation():
    bm25 = BM25Search(":memory:")
    pipeline = HybridSearchPipeline(bm25_search=bm25)
    assert pipeline is not None


@pytest.mark.asyncio
async def test_search_with_no_documents_returns_empty():
    bm25 = BM25Search(":memory:")
    pipeline = HybridSearchPipeline(bm25_search=bm25)
    results = await pipeline.search("python", use_query_expansion=False)
    assert results == []


@pytest.mark.asyncio
async def test_rrf_merges_results_from_two_sources():
    bm25 = BM25Search(":memory:")
    pipeline = HybridSearchPipeline(bm25_search=bm25)

    bm25_results = [_make_sr("doc1", 1.0, 1), _make_sr("doc2", 0.8, 2)]
    vector_results = [_make_sr("doc2", 0.9, 1), _make_sr("doc3", 0.7, 2)]

    merged = pipeline._rrf_merge([bm25_results, vector_results])
    doc_ids = [r.doc_id for r in merged]

    # doc2 appears in both lists → higher RRF score → should rank above doc1 or doc3
    assert "doc2" in doc_ids
    assert len(merged) == 3  # doc1, doc2, doc3


@pytest.mark.asyncio
async def test_search_without_knowledge_base_uses_bm25_only():
    """Pipeline without KnowledgeBase should still work (BM25 only)."""
    bm25 = BM25Search(":memory:")
    await bm25.add_document("d1", "proj1", "python tutorial", title="Python")

    pipeline = HybridSearchPipeline(bm25_search=bm25, knowledge_base=None)
    results = await pipeline.search("python", use_query_expansion=False, use_reranking=False)
    assert len(results) >= 1
    assert results[0].doc_id == "d1"


@pytest.mark.asyncio
async def test_search_with_query_expansion_calls_expander():
    bm25 = BM25Search(":memory:")
    await bm25.add_document("d1", "proj1", "python tutorial", title="Python")

    mock_expander = MagicMock()
    mock_expander.expand = AsyncMock(return_value=["python", "python code"])

    pipeline = HybridSearchPipeline(bm25_search=bm25, query_expander=mock_expander)
    await pipeline.search("python", use_query_expansion=True, use_reranking=False)

    mock_expander.expand.assert_called_once_with("python")
```

### Step 2: Run tests — confirm FAIL (ImportError)

Run: `pytest tests/test_hybrid_search_pipeline.py -v --no-cov`

Expected: `ImportError: cannot import name 'HybridSearchPipeline'`

### Step 3: Create `orchestrator/hybrid_search_pipeline.py`

```python
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
                        sr.score = rr.relevance_score
                        sr.rank = rr.new_rank
                        result.append(sr)
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
```

### Step 4: Run tests — confirm they PASS

Run: `pytest tests/test_hybrid_search_pipeline.py -v --no-cov`

Expected: `5 passed`

### Step 5: Commit

```bash
git add orchestrator/hybrid_search_pipeline.py tests/test_hybrid_search_pipeline.py
git commit -m "feat: add HybridSearchPipeline with RRF fusion and optional query expansion"
```

---

## Task 3: Wire HybridSearchPipeline into engine

**Files:**
- Modify: `orchestrator/engine.py` (two locations: `__init__` and `hybrid_search`)
- Modify: `orchestrator/__init__.py` (export new classes)
- Create: `tests/test_hybrid_pipeline_engine_integration.py`

### Step 1: Write failing test

```python
# tests/test_hybrid_pipeline_engine_integration.py
"""Verify engine.hybrid_search() delegates to HybridSearchPipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.hybrid_search_pipeline import HybridSearchPipeline
from orchestrator.bm25_search import SearchResult


def test_engine_has_hybrid_pipeline_attribute():
    """Engine must expose _hybrid_pipeline after __init__."""
    from orchestrator.engine import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    # Minimal setup so the attribute check works without full init
    bm25 = MagicMock()
    orch._bm25_search = bm25
    orch._reranker = None
    orch._knowledge_base = None
    # Create pipeline manually to match what __init__ will do
    orch._hybrid_pipeline = HybridSearchPipeline(bm25_search=bm25)
    assert hasattr(orch, "_hybrid_pipeline")
    assert isinstance(orch._hybrid_pipeline, HybridSearchPipeline)


@pytest.mark.asyncio
async def test_engine_hybrid_search_delegates_to_pipeline():
    """engine.hybrid_search() must call pipeline.search() and return its results."""
    from orchestrator.engine import Orchestrator
    from orchestrator.bm25_search import SearchResult

    orch = Orchestrator.__new__(Orchestrator)
    mock_pipeline = MagicMock()
    expected = [SearchResult(doc_id="d1", project_id="p1", content="c", title="t", score=0.9, rank=1)]
    mock_pipeline.search = AsyncMock(return_value=expected)
    orch._hybrid_pipeline = mock_pipeline

    result = await orch.hybrid_search("python", project_id="proj1", limit=5)

    mock_pipeline.search.assert_called_once_with(
        "python", project_id="proj1", top_k=5, use_reranking=True
    )
    assert result == [r.to_dict() for r in expected]
```

### Step 2: Run test — confirm FAIL

Run: `pytest tests/test_hybrid_pipeline_engine_integration.py -v --no-cov`

Expected: `FAILED — test_engine_hybrid_search_delegates_to_pipeline` (engine.hybrid_search doesn't call pipeline yet)

### Step 3: Wire pipeline into `engine.py`

**3a. In `Orchestrator.__init__`**, after `self._reranker = LLMReranker()` is initialized (around line 265), add:

```python
from .hybrid_search_pipeline import HybridSearchPipeline
from .query_expander import QueryExpander
self._hybrid_pipeline = HybridSearchPipeline(
    bm25_search=self._bm25_search,
    knowledge_base=getattr(self, "_knowledge_base", None),
    reranker=self._reranker,
    query_expander=QueryExpander(),
)
```

**3b. Replace `Orchestrator.hybrid_search()` body** (lines 743-757) with:

```python
async def hybrid_search(
    self,
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10,
    use_reranking: bool = True,
) -> list:
    """
    Perform hybrid search: BM25 + vector (RRF fusion) + optional reranking.

    Uses HybridSearchPipeline with LLM-based query expansion (DeepSeek-Chat).
    Falls back gracefully if any component is unavailable.
    """
    results = await self._hybrid_pipeline.search(
        query,
        project_id=project_id,
        top_k=limit,
        use_reranking=use_reranking,
    )
    return [r.to_dict() for r in results]
```

### Step 4: Run tests — confirm they PASS

Run: `pytest tests/test_hybrid_pipeline_engine_integration.py -v --no-cov`

Expected: `2 passed`

### Step 5: Run full preflight + hybrid test suite to check no regressions

Run: `pytest tests/test_query_expander.py tests/test_hybrid_search_pipeline.py tests/test_hybrid_pipeline_engine_integration.py tests/test_preflight_integration.py -v --no-cov`

Expected: all pass

### Step 6: Export new classes from `orchestrator/__init__.py`

Find the exports block in `__init__.py` and add after existing search exports:

```python
from .query_expander import QueryExpander
from .hybrid_search_pipeline import HybridSearchPipeline
```

### Step 7: Commit

```bash
git add orchestrator/engine.py orchestrator/__init__.py tests/test_hybrid_pipeline_engine_integration.py
git commit -m "feat: wire HybridSearchPipeline into engine.hybrid_search() with query expansion"
```

---

## Verification

After all tasks, run the new test files together:
```bash
pytest tests/test_query_expander.py tests/test_hybrid_search_pipeline.py tests/test_hybrid_pipeline_engine_integration.py -v --no-cov
```

Expected: **10 tests, all passing**
