"""
Resumed runs and the time budget
================================
A run's time budget is active run time. The saved budget only kept the wall-clock
start, so a run resumed the next day counted the idle gap as spent time and ended
TIMEOUT. And a run that finished every task still ended TIMEOUT or BUDGET_EXHAUSTED
when the limit was crossed on the last task; both statuses are resumable, so
--resume would find nothing to do and stamp the same status again.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.unit

from orchestrator.budget import Budget
from orchestrator.engine_core.state_coordinator import StateCoordinator
from orchestrator.infrastructure.state import _budget_from_dict, _budget_to_dict
from orchestrator.models import Model, ProjectState, ProjectStatus, Task, TaskResult, TaskType

_DAY = 86400.0


def _saved_yesterday(active_seconds: float) -> dict:
    """A budget saved after ``active_seconds`` of running, as stored a day ago."""
    b = Budget(max_time_seconds=5400.0)
    b.original_start_time = time.time() - active_seconds
    d = _budget_to_dict(b)
    d["start_time"] -= _DAY
    d["original_start_time"] -= _DAY
    return d


def test_resume_counts_only_active_time():
    restored = _budget_from_dict(_saved_yesterday(active_seconds=600))

    assert restored.elapsed_seconds == pytest.approx(600, abs=5)
    assert restored.time_remaining()


def test_old_saves_without_elapsed_keep_wall_clock_behaviour():
    d = _saved_yesterday(active_seconds=600)
    d.pop("elapsed_seconds", None)

    restored = _budget_from_dict(d)

    assert restored.elapsed_seconds > _DAY


def _state(budget: Budget, done: int, total: int) -> ProjectState:
    tasks = {f"t{i}": Task(id=f"t{i}", type=TaskType.CODE_GEN, prompt="p") for i in range(total)}
    results = {
        f"t{i}": TaskResult(task_id=f"t{i}", output="ok", score=0.95, model_used=Model.GPT_4O_MINI)
        for i in range(done)
    }
    return ProjectState(
        project_description="d", success_criteria="c", budget=budget, tasks=tasks, results=results
    )


def _over_time() -> Budget:
    b = Budget(max_time_seconds=10.0)
    b.original_start_time = time.time() - 100
    return b


def _spent() -> Budget:
    return Budget(max_usd=1.0, spent_usd=1.0)


@pytest.mark.parametrize("budget", [_over_time(), _spent()], ids=["time", "money"])
def test_finished_run_is_not_reported_as_cut_short(budget):
    status = StateCoordinator().determine_final_status(_state(budget, done=2, total=2))

    assert status == ProjectStatus.SUCCESS


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(_over_time(), ProjectStatus.TIMEOUT), (_spent(), ProjectStatus.BUDGET_EXHAUSTED)],
    ids=["time", "money"],
)
def test_unfinished_run_reports_which_limit_stopped_it(budget, expected):
    status = StateCoordinator().determine_final_status(_state(budget, done=1, total=2))

    assert status == expected


def test_no_results_with_budget_left_is_system_failure():
    status = StateCoordinator().determine_final_status(_state(Budget(), done=0, total=2))

    assert status == ProjectStatus.SYSTEM_FAILURE
