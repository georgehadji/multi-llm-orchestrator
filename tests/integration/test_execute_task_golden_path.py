"""
P0-2: Golden-path canary test for _execute_task.
================================================
If this test breaks during refactoring, stop and diagnose before continuing.

Uses NullState + mock LLM client — no API keys required, no disk I/O.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.budget import Budget
from orchestrator.domain.ports import NullCache, NullState
from orchestrator.engine import Orchestrator
from orchestrator.models import Model, Task, TaskResult, TaskStatus, TaskType


def _inject_mock_client(orch: Orchestrator, mock_client: MagicMock) -> None:
    """Propagate a mock client into all pipeline stages that hold a client ref."""
    # Top-level reference on the orchestrator
    orch.client = mock_client
    # Also patch into the ServiceContainer and any pipeline stages
    if hasattr(orch, "_c") and orch._c is not None:
        orch._c.client = mock_client
    # Walk pipeline stages if accessible
    pipeline = getattr(getattr(orch, "_c", None), "pipeline", None)
    if pipeline is not None:
        for stage in getattr(pipeline, "_stages", []):
            if hasattr(stage, "_client"):
                stage._client = mock_client


@pytest.fixture
async def null_orchestrator():
    """Orchestrator backed by NullCache + NullState with a mock LLM client."""
    orch = Orchestrator(
        budget=Budget(max_usd=5.0, max_time_seconds=300),
        cache=NullCache(),
        state_manager=NullState(),
        max_concurrency=1,
        max_parallel_tasks=1,
    )
    # Replace the real UnifiedClient with a mock that returns a valid CODE_GEN response
    mock_response = MagicMock()
    mock_response.text = "def hello():\n    return 'hello world'"
    mock_response.cost_usd = 0.001
    mock_response.input_tokens = 50
    mock_response.output_tokens = 20
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=mock_response)
    mock_client.is_available = MagicMock(return_value=True)
    _inject_mock_client(orch, mock_client)
    yield orch
    await orch.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_task_returns_completed_result(null_orchestrator):
    """
    Golden path: _execute_task on a CODE_GEN task must return a TaskResult
    with status COMPLETED. This is the primary canary for all Phase 3 extractions.
    """
    orch = null_orchestrator
    task = Task(
        id="canary-001",
        type=TaskType.CODE_GEN,
        prompt="Write a function that returns 'hello world'",
        acceptance_threshold=0.0,  # accept any score so mock evaluator passes
        max_iterations=1,
    )

    result = await orch._execute_task(task)

    assert isinstance(result, TaskResult), "Must return a TaskResult"
    assert result.task_id == "canary-001"
    assert result.status == TaskStatus.COMPLETED, (
        f"Expected COMPLETED, got {result.status}. " f"Output: {result.output[:100]!r}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_task_result_has_output(null_orchestrator):
    """Result output must be non-empty for a successful CODE_GEN task."""
    orch = null_orchestrator
    task = Task(
        id="canary-002",
        type=TaskType.CODE_GEN,
        prompt="Write a hello world function",
        acceptance_threshold=0.0,
        max_iterations=1,
    )

    result = await orch._execute_task(task)

    assert result.output, "output must not be empty on success"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_execute_task_result_has_model(null_orchestrator):
    """Result must record which model was used."""
    orch = null_orchestrator
    task = Task(
        id="canary-003",
        type=TaskType.CODE_GEN,
        prompt="Write a hello world function",
        acceptance_threshold=0.0,
        max_iterations=1,
    )

    result = await orch._execute_task(task)

    assert result.model_used is not None, "model_used must be set"
