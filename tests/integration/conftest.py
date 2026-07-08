"""
Integration test fixtures — reusable mocked Orchestrator with in-memory state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.models import (
    Model,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)
from orchestrator.state import StateManager


@pytest.fixture(autouse=True)
def _bypass_unattended_guard(monkeypatch):
    """Bypass the ENH-4 unattended guard for integration tests.

    These tests exercise orchestration mechanics, not the safety gate itself
    (that is covered by tests/unit/test_unattended_guard.py). Under CI stdin is
    not a TTY, so ``is_unattended`` is True and the guard would otherwise block
    every ``run_project`` call that does not wire a daily cap / retry cap /
    checkpoint. ``ORCH_UNATTENDED_GUARD=false`` is the documented escape hatch.
    """
    monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")


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
async def orchestrator_fixture(temp_state_manager, monkeypatch):
    """
    Provide an Orchestrator with:
      - tiny budget (fast failure if runaway)
      - temp state manager
      - no disk cache (avoid file-system side effects)
      - mocked LLM client (caller must patch client.call or service methods)

    UnifiedClient validates OPENROUTER_API_KEY eagerly at construction; tests
    never make real calls, so provide a dummy key for CI environments.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
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

    # Architecture-rules generation and assumption-surfacing both make real
    # LLM calls before decomposition ever runs, whenever a client is present
    # — regardless of whether the key is real. With a dummy
    # OPENROUTER_API_KEY (see tests/conftest.py) those calls 401 with
    # retries, burning tens of seconds of wall time per test. Skip the
    # architecture-rules phase outright, and neuter network dispatch (not
    # `.call()` itself — that would also skip the circuit-breaker's own
    # open-check, which several tests exercise directly) so any other
    # real-call path fails instantly instead of retrying against the
    # network. Tests exercising one of these phases deliberately can
    # override the relevant mock.
    orch._generate_architecture_rules = AsyncMock(return_value=None)
    orch._c.client._dispatch = AsyncMock(side_effect=RuntimeError("no real network calls in tests"))
    yield orch
    await orch.__aexit__(None, None, None)
