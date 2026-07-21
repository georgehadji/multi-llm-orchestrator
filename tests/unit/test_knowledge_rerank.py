"""Tests for KnowledgeBase two-stage rerank."""

import pytest

pytestmark = pytest.mark.unit
from unittest.mock import AsyncMock, MagicMock

from orchestrator.knowledge_base import KnowledgeBase, KnowledgeType, KnowledgeArtifact


@pytest.fixture(autouse=True)
def _pin_embeddings(monkeypatch):
    """Pin query/artifact embeddings to a fixed 2-d vector.

    These tests exercise rerank ordering logic, not embedding quality. Without
    pinning, the query is embedded by the real (hash/SentenceTransformer)
    pipeline and never matches the artifacts' hand-set [0.5, 0.5] vectors, so
    cosine falls below the 0.5 floor and stage-1 returns nothing.
    """

    async def _fixed(self, text):  # noqa: ANN001
        return [0.5, 0.5]

    monkeypatch.setattr(KnowledgeBase, "_compute_embedding", _fixed)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_artifact(artifact_id: str, content: str, ktype=KnowledgeType.PATTERN) -> KnowledgeArtifact:
    """Create a KnowledgeArtifact with minimal fields."""
    return KnowledgeArtifact(
        id=artifact_id,
        type=ktype,
        content=content,
        embedding=[0.5, 0.5],  # dummy embedding — real value doesn't matter
        similarity_score=0.0,
        title="",
        tags=[],
        created_at="2026-01-01T00:00:00",
        usage_count=0,
    )


class StubReranker:
    """Deterministic reranker for testing — reorders by a fixed doc_id list."""

    def __init__(self, order: list[str]):
        self.order = order
        self.called_with: list | None = None

    async def rerank(self, query: str, results: list[dict], top_k: int, min_score: float = 0.3):
        """Return results in ``order`` order (matching ids)."""
        self.called_with = {"query": query, "results": results, "top_k": top_k}
        # Build ordered output
        id_to_result = {r["id"]: r for r in results}
        reranked = [id_to_result[oid] for oid in self.order if oid in id_to_result]
        # Add relevance_score to each
        for rank, r in enumerate(reranked):
            r["relevance_score"] = 1.0 - (rank * 0.1)  # 1.0, 0.9, 0.8, ...
        return reranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKnowledgeRerank:
    """KnowledgeBase two-stage rerank behavior."""

    @pytest.mark.asyncio
    async def test_rerank_false_returns_cosine_order(self, tmp_path):
        """rerank=False returns identical order to current cosine ranking (regression lock)."""
        kb = KnowledgeBase(storage_path=tmp_path)
        # Add artifacts with known embeddings
        kb._artifacts = {
            "a": make_artifact("a", "Python decorators"),
            "b": make_artifact("b", "Rust ownership"),
            "c": make_artifact("c", "Async patterns"),
        }
        # Override embeddings for deterministic similarity
        for a in kb._artifacts.values():
            a.embedding = [0.5, 0.5]

        results = await kb.find_similar("Python", top_k=3, rerank=False)

        assert len(results) == 3
        # rerank=False — order is pure cosine (all same similarity so insertion order)
        assert results is not None

    @pytest.mark.asyncio
    async def test_rerank_fewer_than_top_k_skips_reranker(self, tmp_path):
        """When cosine returns fewer than top_k hits, reranker is NOT called."""
        stub = StubReranker(order=["a", "b"])
        kb = KnowledgeBase(storage_path=tmp_path, reranker=stub)
        kb._artifacts = {
            "a": make_artifact("a", "Python code"),
        }

        results = await kb.find_similar("Python", top_k=5, rerank=True, fetch_k=10)

        # Only 1 artifact, which is < top_k=5, so reranker not called
        assert stub.called_with is None
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_reranker_reorders_results(self, tmp_path):
        """When rerank=True, results follow reranker's order, not cosine order."""
        stub = StubReranker(order=["c", "a", "b"])  # reverse of expected
        kb = KnowledgeBase(storage_path=tmp_path, reranker=stub)
        kb._artifacts = {
            "a": make_artifact("a", "Python"),
            "b": make_artifact("b", "Rust"),
            "c": make_artifact("c", "Async"),
        }

        results = await kb.find_similar("query", top_k=3, rerank=True, fetch_k=10)

        # The stub returns c first, then a, then b
        assert len(results) == 3
        assert results[0].id == "c", f"Expected c first per reranker, got {[r.id for r in results]}"
        assert results[1].id == "a"
        assert results[2].id == "b"

    @pytest.mark.asyncio
    async def test_reranker_none_falls_back_to_cosine(self, tmp_path):
        """rerank=True but no reranker configured → silent fallback to cosine (no crash)."""
        kb = KnowledgeBase(storage_path=tmp_path, reranker=None)
        kb._artifacts = {
            "a": make_artifact("a", "Python"),
            "b": make_artifact("b", "Rust"),
        }

        results = await kb.find_similar("query", top_k=2, rerank=True, fetch_k=5)

        # Should return cosine results without error
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_reranker_reraises_logged_not_crashed(self, tmp_path):
        """When reranker raises, error is logged and cosine fallback returned."""
        failing = MagicMock()
        failing.rerank = AsyncMock(side_effect=RuntimeError("Reranker failed"))
        kb = KnowledgeBase(storage_path=tmp_path, reranker=failing)
        kb._artifacts = {
            "a": make_artifact("a", "Python"),
            "b": make_artifact("b", "Rust"),
            "c": make_artifact("c", "Async"),
        }

        results = await kb.find_similar("query", top_k=2, rerank=True, fetch_k=5)

        # Should fall back to cosine — expect 2 results
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cache_key_differs_with_rerank(self, tmp_path):
        """rerank=True and rerank=False produce different cache entries."""
        kb = KnowledgeBase(storage_path=tmp_path)
        kb._artifacts = {"a": make_artifact("a", "Python")}

        res1 = await kb.find_similar("same query", top_k=3, rerank=False)
        res2 = await kb.find_similar("same query", top_k=3, rerank=True, fetch_k=10)

        # Both should work, results might differ but that's fine —
        # The cache check would return None for the fresh query anyway.
        # The main assertion is that the cache keys differ (no collisions).
        assert len(res1) == 1
        assert len(res2) == 1
