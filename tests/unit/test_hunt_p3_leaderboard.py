"""Leaderboard — two unread parameters and a cross-module private-attribute read.

Found by the P3 detector, never actioned until this pass converged the root/
analysis/leaderboard.py duplicate pair (S5 of
docs/plans/2026-09-07-patterns-convergence-and-wire-or-delete.md):

- ``get_leaderboard(self, task_type=None)`` never reads ``task_type``. Its two
  call sites are the module-level singleton *factory* (a same-named but
  different function), not this method — there is no caller to preserve a
  filter for.
- ``_get_recommended_tasks(self, model, summary)`` never reads ``model``.
  Private, exactly one caller, which passes both.
- ``pareto_frontier.py`` reads ``leaderboard._summaries`` directly across a
  module boundary — the same encapsulation break that hid the P3
  ``agent._profiles`` bug. ``summary_for()`` replaces it with a public API.
"""

from __future__ import annotations

import inspect

import pytest

from orchestrator.analysis.leaderboard import ModelBenchmarkSummary, ModelLeaderboard
from orchestrator.models import Model


@pytest.fixture
def board(tmp_path) -> ModelLeaderboard:
    return ModelLeaderboard(storage_path=tmp_path / "leaderboard")


@pytest.mark.unit
class TestUnreadParametersAreGone:
    def test_get_leaderboard_has_no_task_type_parameter(self) -> None:
        params = inspect.signature(ModelLeaderboard.get_leaderboard).parameters
        assert "task_type" not in params, (
            "get_leaderboard's task_type parameter is never read in the method "
            "body, and both call sites are the unrelated module-level factory "
            "function -- a parameter that does nothing is silent-absence"
        )

    def test_get_recommended_tasks_has_no_model_parameter(self) -> None:
        params = inspect.signature(ModelLeaderboard._get_recommended_tasks).parameters
        assert "model" not in params, (
            "_get_recommended_tasks' model parameter is never read in the "
            "method body; its one caller passes it for nothing"
        )

    def test_get_leaderboard_still_ranks_by_composite_score(self, board) -> None:
        """Behaviour-preserving: dropping the dead parameter must not touch ranking."""
        good = ModelBenchmarkSummary(model=Model.GPT_4O_MINI, avg_quality=0.9, benchmark_count=5)
        bad = ModelBenchmarkSummary(model=Model.GEMINI_FLASH, avg_quality=0.2, benchmark_count=5)
        board._summaries[Model.GPT_4O_MINI] = good
        board._summaries[Model.GEMINI_FLASH] = bad

        entries = board.get_leaderboard()

        assert [e.model for e in entries] == [Model.GPT_4O_MINI, Model.GEMINI_FLASH]
        assert entries[0].rank == 1


@pytest.mark.unit
class TestSummaryForAccessor:
    def test_summary_for_returns_the_stored_summary(self, board) -> None:
        summary = ModelBenchmarkSummary(model=Model.GPT_4O_MINI, avg_quality=0.8)
        board._summaries[Model.GPT_4O_MINI] = summary

        assert board.summary_for(Model.GPT_4O_MINI) is summary

    def test_summary_for_returns_none_when_absent(self, board) -> None:
        assert board.summary_for(Model.GPT_4O_MINI) is None
