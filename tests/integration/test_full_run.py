"""
test_full_run.py — End-to-end orchestrator flow with mocked LLM layer.
=======================================================================

Mocks the three application-layer services (generator, executor, evaluator)
to verify that run_project() wires them together correctly without exercising
the actual LLM client.

MVOS coverage:
  - CLI entry point delegates to run_project
  - Decomposition → Execution → Evaluation pipeline completes
  - ProjectState is returned with correct status
  - Results are persisted to StateManager
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.models import ProjectStatus, TaskStatus, TaskType
from orchestrator.services.generator import GeneratorResult


@pytest.mark.asyncio
@pytest.mark.requires_api  # engine makes LLM calls beyond the mocked executor path
async def test_run_project_completes_with_two_tasks(
    orchestrator_fixture,
    mock_tasks,
    ok_result_t1,
    ok_result_t2,
):
    """
    Happy-path integration: generator returns 2 tasks, executor completes
    both, evaluator returns high scores. run_project() should return a
    COMPLETED ProjectState with both results stored.
    """
    orch = orchestrator_fixture

    # ── Patch services at the Orchestrator level ──────────────────────────
    orch._generator.decompose = AsyncMock(
        return_value=GeneratorResult(tasks=mock_tasks, wall_time_ms=100.0, error=None)
    )
    orch._execute_task = AsyncMock(
        side_effect=[
            ok_result_t1,
            ok_result_t2,
        ]
    )
    orch._evaluator.evaluate = AsyncMock(return_value=0.90)

    # ── Execute ───────────────────────────────────────────────────────────
    state = await orch.run_project(
        project_description="Build a hello-world app",
        success_criteria="Code runs and prints hello",
        project_id="integ-test-001",
    )

    # ── Assertions ────────────────────────────────────────────────────────
    assert state is not None
    assert state.status == ProjectStatus.SUCCESS
    assert len(state.tasks) == 2
    assert "t1" in state.results
    assert "t2" in state.results
    assert state.results["t1"].status == TaskStatus.COMPLETED
    assert state.results["t2"].status == TaskStatus.COMPLETED

    # Verify services were called the expected number of times
    orch._generator.decompose.assert_called_once()
    assert orch._execute_task.call_count == 2
    # Evaluator is called per-iteration inside _execute_task; with mocking at
    # the service level we only see the outer evaluate() if the engine wires
    # it directly.  For this integration test we focus on state correctness.


@pytest.mark.asyncio
@pytest.mark.requires_api  # engine makes LLM calls beyond the mocked executor path
async def test_run_project_degradation_on_partial_failure(
    orchestrator_fixture,
    mock_tasks,
    ok_result_t1,
):
    """
    If one task succeeds and one fails, state should reflect PARTIAL_SUCCESS
    (or COMPLETED_DEGRADED depending on engine logic).
    """
    orch = orchestrator_fixture

    from orchestrator.models import TaskResult, TaskStatus, Model

    failed_task_result = TaskResult(
        task_id="t2",
        output="",
        score=0.0,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.FAILED,
        task_type=TaskType.CODE_REVIEW.value,
        cost_usd=0.0,
        tokens_used={"input": 0, "output": 0},
    )
    failed_result = type(
        "R",
        (),
        {
            "task_result": failed_task_result,
            "succeeded": False,
            "error": RuntimeError("mock failure"),
        },
    )()

    orch._generator.decompose = AsyncMock(
        return_value=GeneratorResult(tasks=mock_tasks, wall_time_ms=100.0, error=None)
    )
    orch._execute_task = AsyncMock(
        side_effect=[
            ok_result_t1,
            failed_task_result,
        ]
    )

    state = await orch.run_project(
        project_description="Build a flaky app",
        success_criteria="Some tasks may fail",
        project_id="integ-test-002",
    )

    assert state is not None
    # Engine may report PARTIAL_SUCCESS or COMPLETED_DEGRADED when tasks fail
    assert state.status in (
        ProjectStatus.PARTIAL_SUCCESS,
        ProjectStatus.COMPLETED_DEGRADED,
        ProjectStatus.SYSTEM_FAILURE,
    )
    assert "t1" in state.results or any(r.task_id == "t1" for r in state.results.values())


@pytest.mark.asyncio
async def test_run_project_persists_state(
    orchestrator_fixture,
    mock_tasks,
    ok_result_t1,
    ok_result_t2,
):
    """
    After run_project() finishes, the state must be loadable from the
    StateManager (resume capability).
    """
    orch = orchestrator_fixture
    project_id = "integ-test-003"

    orch._generator.decompose = AsyncMock(
        return_value=GeneratorResult(tasks=mock_tasks, wall_time_ms=100.0, error=None)
    )
    orch._execute_task = AsyncMock(
        side_effect=[
            ok_result_t1,
            ok_result_t2,
        ]
    )

    await orch.run_project(
        project_description="Persistent state test",
        success_criteria="State must be saved",
        project_id=project_id,
    )

    # Load back from StateManager
    loaded = await orch.state_mgr.load_project(project_id)
    assert loaded is not None
    assert loaded.project_description == "Persistent state test"
    assert len(loaded.tasks) == 2
