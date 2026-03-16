"""
Regression tests for three silent runtime bugs fixed in v5.1.

BUG-001  engine.py   Budget reservation leaked on run_project() failure
BUG-002  engine.py   asyncio.gather without return_exceptions left orphan tasks
BUG-003  hybrid_search_pipeline.py  SearchResult mutated in-place during reranking
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# BUG-001 — Budget reservation not released when run_project() fails
# ---------------------------------------------------------------------------

from orchestrator.cost import BudgetHierarchy


class TestBudgetReservationReleasedOnFailure:
    """BudgetHierarchy reservation must be zeroed out when the job never charges."""

    def _hierarchy(self) -> BudgetHierarchy:
        return BudgetHierarchy(
            org_max_usd=100.0,
            team_budgets={"eng": 50.0},
        )

    def test_reservation_released_after_release_reservation(self):
        """release_reservation() must free both org and team counters."""
        h = self._hierarchy()
        assert h.can_afford_job("job-1", "eng", 30.0) is True
        # Reservation committed — org and team reserved should be 30
        assert h._reserved_usd == pytest.approx(30.0)
        assert h._team_reserved.get("eng", 0.0) == pytest.approx(30.0)

        h.release_reservation("job-1", "eng")

        assert h._reserved_usd == pytest.approx(0.0), "org reservation must be freed"
        assert h._team_reserved.get("eng", 0.0) == pytest.approx(
            0.0
        ), "team reservation must be freed"

    def test_reservation_released_is_idempotent(self):
        """Calling release_reservation twice must not go negative."""
        h = self._hierarchy()
        h.can_afford_job("job-2", "eng", 20.0)
        h.release_reservation("job-2", "eng")
        h.release_reservation("job-2", "eng")  # second call — must not raise or go negative
        assert h._reserved_usd == pytest.approx(0.0)

    def test_next_job_can_use_released_budget(self):
        """After reservation release, a subsequent job must be approvable."""
        h = self._hierarchy()
        assert h.can_afford_job("job-3", "eng", 40.0) is True
        # Simulate failure — release without charging
        h.release_reservation("job-3", "eng")
        # Budget is fully restored; a new job for the same amount should pass
        assert h.can_afford_job("job-4", "eng", 40.0) is True

    def test_charge_without_release_succeeds(self):
        """charge_job() called after can_afford_job() must still work correctly."""
        h = self._hierarchy()
        h.can_afford_job("job-5", "eng", 25.0)
        h.charge_job("job-5", "eng", 25.0)
        assert h._reserved_usd == pytest.approx(0.0)
        assert h._org_spent == pytest.approx(25.0)
        remaining = h.remaining("org")
        assert remaining == pytest.approx(75.0)

    @pytest.mark.asyncio
    async def test_run_job_releases_reservation_on_failure(self):
        """engine.run_job() must release the BudgetHierarchy reservation when
        run_project() raises, so the org budget is not permanently locked."""
        from orchestrator.cost import BudgetHierarchy
        from orchestrator.policy import JobSpec, PolicySet
        from orchestrator.models import Budget

        hierarchy = BudgetHierarchy(org_max_usd=100.0)

        # Build a minimal Orchestrator with the hierarchy wired in
        with patch("orchestrator.engine.Orchestrator.__init__", return_value=None):
            from orchestrator.engine import Orchestrator

            orch = object.__new__(Orchestrator)
            orch._budget_hierarchy = hierarchy
            orch.budget = Budget(max_usd=10.0)
            orch._active_policies = PolicySet()
            orch._max_parallel_tasks = 1

            # Stub out helpers that run_job() calls before/after run_project()
            orch._apply_warm_start = AsyncMock()
            orch._flush_telemetry_snapshots = AsyncMock()

            # run_project() always raises
            orch.run_project = AsyncMock(side_effect=RuntimeError("simulated failure"))

            spec = MagicMock()
            spec.job_id = "fail-job"
            spec.team = ""
            spec.budget = Budget(max_usd=50.0)
            spec.project_description = "test"
            spec.success_criteria = "ok"
            spec.max_parallel_tasks = 0
            spec.policy_set = PolicySet()

            with pytest.raises(RuntimeError, match="simulated failure"):
                await orch.run_job(spec)

        # Reservation must have been released
        assert hierarchy._reserved_usd == pytest.approx(
            0.0
        ), "org reservation leaked after run_project() failure"


# ---------------------------------------------------------------------------
# BUG-002 — asyncio.gather without return_exceptions left orphan tasks
# ---------------------------------------------------------------------------


class TestGatherReturnExceptions:
    """Level execution must use return_exceptions=True so all tasks in a
    level complete before the checkpoint snapshot is taken."""

    @pytest.mark.asyncio
    async def test_all_tasks_complete_when_one_raises(self):
        """With return_exceptions=True, a failing task must not prevent its
        siblings from completing."""
        completed: list[str] = []

        async def good_task(name: str) -> None:
            await asyncio.sleep(0)
            completed.append(name)

        async def bad_task() -> None:
            await asyncio.sleep(0)
            raise ValueError("deliberate")

        results = await asyncio.gather(
            good_task("A"),
            bad_task(),
            good_task("B"),
            return_exceptions=True,
        )

        assert "A" in completed, "task A must complete even though sibling raised"
        assert "B" in completed, "task B must complete even though sibling raised"
        exceptions = [r for r in results if isinstance(r, BaseException)]
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], ValueError)

    @pytest.mark.asyncio
    async def test_results_dict_consistent_after_partial_failure(self):
        """Simulates the level-execution loop: self.results must contain
        entries for all tasks in the level, not just those before the failure."""
        results_dict: dict[str, str] = {}

        async def run_task(tid: str, should_fail: bool) -> None:
            await asyncio.sleep(0)
            if should_fail:
                raise RuntimeError(f"{tid} failed")
            results_dict[tid] = "done"

        level_results = await asyncio.gather(
            run_task("t1", False),
            run_task("t2", True),
            run_task("t3", False),
            return_exceptions=True,
        )

        # All non-failing tasks should have written their results
        assert results_dict.get("t1") == "done"
        assert results_dict.get("t3") == "done"
        # t2 failed — ensure its exception surfaced (not silently lost)
        exceptions = [r for r in level_results if isinstance(r, BaseException)]
        assert any("t2 failed" in str(e) for e in exceptions)


# ---------------------------------------------------------------------------
# BUG-003 — SearchResult mutated in-place during reranking
# ---------------------------------------------------------------------------

from orchestrator.bm25_search import SearchResult
from orchestrator.hybrid_search_pipeline import HybridSearchPipeline


def _make_sr(doc_id: str, score: float, rank: int) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        project_id="proj",
        content=f"content of {doc_id}",
        title=f"title {doc_id}",
        score=score,
        rank=rank,
    )


class TestSearchResultNotMutatedDuringReranking:
    """fused SearchResult objects must not be mutated in-place by reranking."""

    @pytest.mark.asyncio
    async def test_fused_objects_unchanged_after_successful_reranking(self):
        """After reranking, the original fused list must preserve its RRF scores."""

        @dataclass
        class FakeRerankResult:
            doc_id: str
            relevance_score: float
            new_rank: int

        async def fake_rerank(query: str, dicts: list, top_k: int = 10):
            return [
                FakeRerankResult(doc_id="d1", relevance_score=0.9, new_rank=1),
                FakeRerankResult(doc_id="d2", relevance_score=0.7, new_rank=2),
            ]

        bm25 = MagicMock()
        bm25.bm25_search = AsyncMock(return_value=[
            _make_sr("d1", score=0.5, rank=1),
            _make_sr("d2", score=0.4, rank=2),
        ])
        reranker = MagicMock()
        reranker.rerank = fake_rerank

        pipeline = HybridSearchPipeline(bm25_search=bm25, reranker=reranker)
        results = await pipeline.search(
            "query", use_query_expansion=False, use_reranking=True
        )

        # Results must reflect reranker scores
        assert results[0].doc_id == "d1"
        assert results[0].score == pytest.approx(0.9)
        assert results[0].rank == 1

        # The BM25 source object must NOT have been mutated
        original_sr = (await bm25.bm25_search("query", project_id=None, limit=30))[0]
        assert original_sr.score == pytest.approx(0.5), "BM25 source object was mutated"
        assert original_sr.rank == 1

    @pytest.mark.asyncio
    async def test_fused_fallback_unmodified_when_reranking_raises_mid_loop(self):
        """If reranking raises after partially processing results, the fallback
        fused[:top_k] must contain unmodified RRF scores — not a corrupted mix."""

        call_count = 0

        @dataclass
        class FakeRerankResult:
            doc_id: str
            relevance_score: float
            new_rank: int

        async def partial_fail_rerank(query: str, dicts: list, top_k: int = 10):
            # Raises before returning anything
            raise TimeoutError("LLM timeout mid-rerank")

        bm25 = MagicMock()
        rrf_results = [
            _make_sr("d1", score=0.6, rank=1),
            _make_sr("d2", score=0.4, rank=2),
            _make_sr("d3", score=0.2, rank=3),
        ]
        bm25.bm25_search = AsyncMock(return_value=rrf_results)
        reranker = MagicMock()
        reranker.rerank = partial_fail_rerank

        pipeline = HybridSearchPipeline(bm25_search=bm25, reranker=reranker)
        results = await pipeline.search(
            "query", use_query_expansion=False, use_reranking=True, top_k=3
        )

        # Must fall back to RRF results, all with original RRF scores
        assert len(results) == 3
        scores = {r.doc_id: r.score for r in results}
        # No result should have a reranker score (0.9, 0.7, etc.) or rank 0
        assert all(r.rank > 0 for r in results), "rank must never be 0 in fallback"
        assert scores["d1"] == pytest.approx(
            results[0].score
        ), "fallback results must be consistent RRF scores"

    @pytest.mark.asyncio
    async def test_reranked_objects_are_new_instances(self):
        """Returned SearchResult objects must be distinct instances from fused."""

        @dataclass
        class FakeRerankResult:
            doc_id: str
            relevance_score: float
            new_rank: int

        async def fake_rerank(query: str, dicts: list, top_k: int = 10):
            return [FakeRerankResult(doc_id="d1", relevance_score=0.99, new_rank=1)]

        original = _make_sr("d1", score=0.5, rank=1)
        bm25 = MagicMock()
        bm25.bm25_search = AsyncMock(return_value=[original])
        reranker = MagicMock()
        reranker.rerank = fake_rerank

        pipeline = HybridSearchPipeline(bm25_search=bm25, reranker=reranker)
        results = await pipeline.search(
            "q", use_query_expansion=False, use_reranking=True
        )

        assert results[0] is not original, "reranked result must be a new object"
        assert original.score == pytest.approx(0.5), "original must be unmodified"
        assert results[0].score == pytest.approx(0.99)
