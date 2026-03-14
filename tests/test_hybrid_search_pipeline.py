"""Tests for HybridSearchPipeline — BM25 + Vector + RRF + Query Expansion."""
import pytest
from unittest.mock import AsyncMock, MagicMock
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
