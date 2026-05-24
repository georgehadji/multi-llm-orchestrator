"""
Integration test fixtures — reusable mocked Orchestrator with in-memory state.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.models import (
    Model,
    ProjectState,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)
from orchestrator.state import StateManager


@pytest.fixture
async def temp_state_manager():
    """Provide a StateManager backed by a temporary SQLite DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    sm = StateManager(db_path=db_path)
    yield sm
    await sm.close()


@pytest.fixture
def mock_task() -> Task:
    """Single generic code-gen task."""
    return Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="Write hello world",
        context="",
        dependencies=[],
    )


@pytest.fixture
def mock_tasks() -> dict[str, Task]:
    """Two-task dict for decomposition return."""
    return {
        "t1": Task(
            id="t1",
            type=TaskType.CODE_GEN,
            prompt="Write hello world",
            context="",
            dependencies=[],
        ),
        "t2": Task(
            id="t2",
            type=TaskType.CODE_REVIEW,
            prompt="Review the hello world code",
            context="",
            dependencies=["t1"],
        ),
    }


@pytest.fixture
def ok_result_t1() -> TaskResult:
    return TaskResult(
        task_id="t1",
        output="print('hello world')",
        score=0.92,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.COMPLETED,
        task_type=TaskType.CODE_GEN.value,
        critique="",
        iterations=1,
        cost_usd=0.001,
        tokens_used={"input": 10, "output": 5},
    )


@pytest.fixture
def ok_result_t2() -> TaskResult:
    return TaskResult(
        task_id="t2",
        output="Code looks good",
        score=0.88,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.COMPLETED,
        task_type=TaskType.CODE_REVIEW.value,
        critique="",
        iterations=1,
        cost_usd=0.001,
        tokens_used={"input": 8, "output": 4},
    )


@pytest.fixture
async def orchestrator_fixture(temp_state_manager):
    """
    Provide an Orchestrator with:
      - tiny budget (fast failure if runaway)
      - temp state manager
      - no disk cache (avoid file-system side effects)
      - mocked LLM client (caller must patch client.call or service methods)
    """
    orch = Orchestrator(
        budget=Budget(max_usd=1.0, max_time_seconds=300),
        state_manager=temp_state_manager,
        max_concurrency=1,
        max_parallel_tasks=1,
    )
    # Disable cache to avoid cross-test file pollution
    orch.cache = MagicMock()
    orch.cache.get = AsyncMock(return_value=None)
    orch.cache.put = AsyncMock(return_value=None)
    orch.cache.close = AsyncMock(return_value=None)
    yield orch
    await orch.__aexit__(None, None, None)
