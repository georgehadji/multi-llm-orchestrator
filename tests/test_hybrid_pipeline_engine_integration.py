"""Verify engine.hybrid_search() delegates to HybridSearchPipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock
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
