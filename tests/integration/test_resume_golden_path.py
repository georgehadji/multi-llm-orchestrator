"""
P0-3: Golden-path canary test for project resume (_resume_project).
====================================================================
Pre-populates NullState with a PARTIAL_SUCCESS ProjectState that has one
completed task and one failed task, then asserts run_project re-enters the
resume path and does not return SYSTEM_FAILURE.

No API keys required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration
from unittest.mock import AsyncMock, MagicMock

from orchestrator.budget import Budget
from orchestrator.domain.ports import NullCache, NullState
from orchestrator.engine import Orchestrator
from orchestrator.models import (
    Model,
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)


def _inject_mock_client(orch: Orchestrator, mock_client: MagicMock) -> None:
    """Propagate a mock client into all pipeline stages that hold a client ref."""
    orch.client = mock_client
    if hasattr(orch, "_c") and orch._c is not None:
        orch._c.client = mock_client
    pipeline = getattr(getattr(orch, "_c", None), "pipeline", None)
    if pipeline is not None:
        for stage in getattr(pipeline, "_stages", []):
            if hasattr(stage, "_client"):
                stage._client = mock_client


def _make_partial_state() -> tuple[str, ProjectState]:
    """Build a PARTIAL_SUCCESS ProjectState with one done + one failed task."""
    project_id = "resume-canary-001"
    tasks = {
        "t1": Task(
            id="t1",
            type=TaskType.CODE_GEN,
            prompt="Write hello world",
            acceptance_threshold=0.0,
            max_iterations=1,
        ),
        "t2": Task(
            id="t2",
            type=TaskType.CODE_GEN,
            prompt="Write goodbye world",
            acceptance_threshold=0.0,
            max_iterations=1,
        ),
    }
    results = {
        "t1": TaskResult(
            task_id="t1",
            output="print('hello world')",
            score=0.9,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            task_type=TaskType.CODE_GEN.value,
            iterations=1,
            cost_usd=0.001,
        ),
        "t2": TaskResult(
            task_id="t2",
            output="",
            score=0.0,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.FAILED,
            task_type=TaskType.CODE_GEN.value,
            iterations=1,
            cost_usd=0.0,
        ),
    }
    state = ProjectState(
        project_description="Build hello/goodbye app",
        success_criteria="Both functions exist",
        budget=Budget(max_usd=5.0, max_time_seconds=300),
        tasks=tasks,
        results=results,
        status=ProjectStatus.PARTIAL_SUCCESS,
        execution_order=["t1", "t2"],
    )
    return project_id, state


@pytest.fixture
async def resume_orchestrator():
    """Orchestrator with a pre-seeded PARTIAL_SUCCESS state."""
    project_id, partial_state = _make_partial_state()
    null_state = NullState()
    await null_state.save_project(project_id, partial_state)

    orch = Orchestrator(
        budget=Budget(max_usd=5.0, max_time_seconds=300),
        cache=NullCache(),
        state_manager=null_state,
        max_concurrency=1,
        max_parallel_tasks=1,
    )
    mock_response = MagicMock()
    mock_response.text = "print('goodbye world')"
    mock_response.cost_usd = 0.001
    mock_response.input_tokens = 30
    mock_response.output_tokens = 10
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=mock_response)
    mock_client.is_available = MagicMock(return_value=True)
    _inject_mock_client(orch, mock_client)

    yield orch, project_id
    await orch.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_resume_does_not_return_system_failure(resume_orchestrator):
    """
    When run_project finds a PARTIAL_SUCCESS checkpoint it must resume,
    not return SYSTEM_FAILURE. This is the resume-path canary.
    """
    orch, project_id = resume_orchestrator

    state = await orch.run_project(
        project_description="Build hello/goodbye app",
        success_criteria="Both functions exist",
        project_id=project_id,
    )

    assert (
        state.status != ProjectStatus.SYSTEM_FAILURE
    ), f"Resume must not return SYSTEM_FAILURE, got {state.status}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_resume_preserves_completed_task_result(resume_orchestrator):
    """
    The already-completed task (t1) must appear in the final state with
    COMPLETED status — resume must not re-execute finished tasks.
    """
    orch, project_id = resume_orchestrator

    state = await orch.run_project(
        project_description="Build hello/goodbye app",
        success_criteria="Both functions exist",
        project_id=project_id,
    )

    assert "t1" in state.results, "t1 must be in results after resume"
    assert (
        state.results["t1"].status == TaskStatus.COMPLETED
    ), "Completed task t1 must remain COMPLETED after resume"
