"""
Integration tests for orchestrator.application.resumption_service.ResumptionService

P3-3 of REFACTORING_PLAN_V7.md — extracted from engine._resume_project.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.budget import Budget
from orchestrator.models import (
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)


pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_task(task_id: str = "t1") -> Task:
    t = MagicMock(spec=Task)
    t.id = task_id
    t.type = TaskType.CODE_GEN
    return t


def _make_result(status: TaskStatus = TaskStatus.COMPLETED) -> TaskResult:
    r = MagicMock(spec=TaskResult)
    r.status = status
    return r


def _make_budget(spent: float = 0.5) -> Budget:
    b = Budget(max_usd=10.0)
    b.spent_usd = spent
    b.phase_spent = {"generation": spent}
    b.original_start_time = time.time() - 3600  # 1 hour ago
    return b


def _make_state(
    tasks: dict | None = None,
    results: dict | None = None,
    execution_order: list | None = None,
    budget_spent: float = 0.5,
) -> ProjectState:
    tasks = tasks or {}
    results = results or {}
    execution_order = execution_order or list(tasks.keys())
    b = _make_budget(budget_spent)
    return ProjectState(
        project_description="test project",
        success_criteria="tests pass",
        budget=b,
        tasks=tasks,
        results=results,
        status=ProjectStatus.PARTIAL_SUCCESS,
        execution_order=execution_order,
    )


def _make_service(execute_task_fn=None, determine_status_fn=None):
    from orchestrator.application.resumption_service import ResumptionService

    budget = Budget(max_usd=10.0)
    results: dict = {}

    if execute_task_fn is None:
        execute_task_fn = AsyncMock(return_value=_make_result())
    if determine_status_fn is None:
        determine_status_fn = MagicMock(return_value=ProjectStatus.SUCCESS)

    svc = ResumptionService(
        budget=budget,
        results=results,
        execute_task_fn=execute_task_fn,
        determine_final_status_fn=determine_status_fn,
    )
    return svc, budget, results, execute_task_fn, determine_status_fn


# ─────────────────────────────────────────────────────────────────────────────
# Budget restoration
# ─────────────────────────────────────────────────────────────────────────────


async def test_resume_restores_budget_spent():
    svc, budget, _, _, _ = _make_service()
    state = _make_state(budget_spent=1.23)

    await svc.resume(state)

    assert budget.spent_usd == pytest.approx(1.23)


async def test_resume_restores_phase_spent():
    svc, budget, _, _, _ = _make_service()
    state = _make_state(budget_spent=0.5)
    state.budget.phase_spent = {"generation": 0.5, "evaluation": 0.1}

    await svc.resume(state)

    assert budget.phase_spent["generation"] == pytest.approx(0.5)
    assert budget.phase_spent["evaluation"] == pytest.approx(0.1)


async def test_resume_restores_original_start_time():
    svc, budget, _, _, _ = _make_service()
    original_ts = time.time() - 7200
    state = _make_state()
    state.budget.original_start_time = original_ts

    await svc.resume(state)

    assert budget.original_start_time == pytest.approx(original_ts)


async def test_resume_resets_start_time_to_now():
    svc, budget, _, _, _ = _make_service()
    before = time.time()
    state = _make_state()
    # Force old start_time so we can verify it was reset
    state.budget.start_time = before - 9999

    await svc.resume(state)

    assert budget.start_time >= before


# ─────────────────────────────────────────────────────────────────────────────
# Task execution
# ─────────────────────────────────────────────────────────────────────────────


async def test_resume_copies_existing_results():
    t1 = _make_task("t1")
    r1 = _make_result(TaskStatus.COMPLETED)
    svc, _, results, _, _ = _make_service()
    state = _make_state(tasks={"t1": t1}, results={"t1": r1}, execution_order=["t1"])

    await svc.resume(state)

    assert "t1" in results


async def test_resume_executes_pending_tasks():
    t1 = _make_task("t1")
    execute_fn = AsyncMock(return_value=_make_result(TaskStatus.COMPLETED))  # type: ignore[arg-type]
    svc, _, results, execute_fn, _ = _make_service(execute_task_fn=execute_fn)
    state = _make_state(tasks={"t1": t1}, results={}, execution_order=["t1"])

    await svc.resume(state)

    execute_fn.assert_awaited_once()
    assert "t1" in results


async def test_resume_executes_failed_tasks():
    t1 = _make_task("t1")
    r1 = _make_result(TaskStatus.FAILED)
    execute_fn = AsyncMock(return_value=_make_result(TaskStatus.COMPLETED))  # type: ignore[arg-type]
    svc, _, results, execute_fn, _ = _make_service(execute_task_fn=execute_fn)
    state = _make_state(tasks={"t1": t1}, results={"t1": r1}, execution_order=["t1"])

    await svc.resume(state)

    execute_fn.assert_awaited_once()


async def test_resume_skips_completed_tasks():
    t1 = _make_task("t1")
    r1 = _make_result(TaskStatus.COMPLETED)
    execute_fn = AsyncMock(return_value=_make_result(TaskStatus.COMPLETED))  # type: ignore[arg-type]
    svc, _, _, execute_fn, _ = _make_service(execute_task_fn=execute_fn)
    state = _make_state(tasks={"t1": t1}, results={"t1": r1}, execution_order=["t1"])

    await svc.resume(state)

    execute_fn.assert_not_awaited()


async def test_resume_updates_state_results():
    t1 = _make_task("t1")
    new_result = _make_result(TaskStatus.COMPLETED)
    execute_fn = AsyncMock(return_value=new_result)
    svc, _, _, _, _ = _make_service(execute_task_fn=execute_fn)
    state = _make_state(tasks={"t1": t1}, results={}, execution_order=["t1"])

    # M7: resume returns a NEW state (immutable); original is unchanged
    returned_state = await svc.resume(state)

    assert returned_state.results["t1"] is new_result


# ─────────────────────────────────────────────────────────────────────────────
# Final status
# ─────────────────────────────────────────────────────────────────────────────


async def test_resume_sets_final_status_from_callback():
    svc, _, _, _, determine_fn = _make_service(
        determine_status_fn=MagicMock(return_value=ProjectStatus.SUCCESS)
    )
    state = _make_state()

    result = await svc.resume(state)

    assert result.status == ProjectStatus.SUCCESS
    determine_fn.assert_called_once_with(state)


async def test_resume_returns_new_state():
    """M7: resume must return a new ProjectState, not the input mutated in place."""
    svc, _, _, _, _ = _make_service()
    state = _make_state()

    returned = await svc.resume(state)

    # Should be a distinct object (immutable pattern)
    assert returned is not state
    # But should be a ProjectState with the correct type
    from orchestrator.models import ProjectState
    assert isinstance(returned, ProjectState)
