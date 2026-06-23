"""
Shared pytest fixtures for the AI Orchestrator test suite.

Provides reusable fixtures for Budget, Task, CircuitBreaker,
temp state managers, mock clients, and common test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.application.model_profile_builder import build_default_profiles
from orchestrator.models import (
    Budget,
    Model,
    Task,
    TaskType,
    TaskStatus,
    TaskResult,
)
from orchestrator.circuit_breaker import CircuitBreaker

# ═══════════════════════════════════════════════════════════════════════════
# Budget Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_budget() -> Budget:
    """A Budget with $5.00 limit, 300s timeout."""
    return Budget(max_usd=5.0, max_time_seconds=300)


@pytest.fixture
def large_budget() -> Budget:
    """A Budget with $100.00 limit, 3600s timeout."""
    return Budget(max_usd=100.0, max_time_seconds=3600)


# ═══════════════════════════════════════════════════════════════════════════
# Task Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def code_task() -> Task:
    """A basic code generation task."""
    return Task(
        id="task_001",
        type=TaskType.CODE_GEN,
        prompt="Write a FastAPI health endpoint",
        target_path="src/api/health.py",
    )


@pytest.fixture
def review_task() -> Task:
    """A code review task."""
    return Task(
        id="task_002",
        type=TaskType.CODE_REVIEW,
        prompt="Review the auth module",
        dependencies=["task_001"],
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    """A list of 3 sample tasks with dependencies."""
    return [
        Task(id="task_001", type=TaskType.CODE_GEN, prompt="Setup project structure"),
        Task(
            id="task_002",
            type=TaskType.CODE_GEN,
            prompt="Add auth module",
            dependencies=["task_001"],
        ),
        Task(
            id="task_003",
            type=TaskType.TEST_GEN,
            prompt="Write tests for auth",
            dependencies=["task_002"],
        ),
    ]


@pytest.fixture
def task_result_ok() -> TaskResult:
    """A successful TaskResult."""
    return TaskResult(
        task_id="task_001",
        status=TaskStatus.COMPLETED,
        output="def health(): return {'status': 'ok'}",
        score=0.95,
        cost_usd=0.01,
        model_used=Model.GPT_4O,
        iterations=2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """A CircuitBreaker with low thresholds for testing."""
    return CircuitBreaker(
        name="test-breaker",
        failure_threshold=2,
        reset_timeout=0.05,
        success_threshold=2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Temp Directory Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory that persists for the test session."""
    return tmp_path


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Temporary project directory with some files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("# Test Project")
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════
# Mock Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock LLM client that returns a predefined response."""
    client = MagicMock()
    client.call = AsyncMock()
    response = MagicMock()
    response.text = '{"score": 0.85, "issues": []}'
    response.cost_usd = 0.01
    response.input_tokens = 100
    response.output_tokens = 50
    client.call.return_value = response
    return client


@pytest.fixture
def mock_telemetry() -> MagicMock:
    """Mock TelemetryCollector."""
    return MagicMock()


@pytest.fixture
def mock_state_manager() -> MagicMock:
    """Mock StateManager with async methods."""
    manager = MagicMock()
    manager.save_project = AsyncMock(return_value=True)
    manager.load_project = AsyncMock(return_value=None)
    manager.save_checkpoint = AsyncMock()
    manager.close = AsyncMock()
    return manager


# ═══════════════════════════════════════════════════════════════════════════
# Profile Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def default_profiles() -> dict[Model, Any]:
    """Default model profiles for testing."""
    return build_default_profiles()


# ═══════════════════════════════════════════════════════════════════════════
# Async Helpers
# ═══════════════════════════════════════════════════════════════════════════
# NOTE: Do NOT override the event_loop fixture here.
# pytest-asyncio >= 0.23 removed support for custom event_loop overrides.
# Loop scope is configured via asyncio_default_fixture_loop_scope in pyproject.toml.


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (single function/method)")
    config.addinivalue_line("markers", "integration: Integration tests (multi-module)")
    config.addinivalue_line("markers", "slow: Slow tests (> 1 second)")
    config.addinivalue_line("markers", "requires_api: Tests requiring API keys")
    config.addinivalue_line("markers", "edge_case: Edge case and boundary tests")
