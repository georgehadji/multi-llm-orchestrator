"""
Characterization tests for orchestrator.engine.Orchestrator.run_project()

These tests capture the current behavior of the engine's main entry point as a golden master.
They serve two purposes:
1. Document the expected I/O contract before refactoring (run_project signature and ProjectState output)
2. Catch regressions during engine decomposition (phases C1-C7)

Tests mock ServiceContainer.build() to avoid external API calls and I/O.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.models import ProjectState, Budget as BudgetModel


@pytest.fixture
def mock_container():
    """Build a mocked ServiceContainer that prevents real I/O."""
    from orchestrator.models import Model

    container = MagicMock()
    container.client = MagicMock()
    container.client.is_available.return_value = True
    container.budget = Budget(max_usd=10.0, max_time_seconds=300)
    container.cache = MagicMock()
    container.state_mgr = MagicMock()
    container.pipeline_runner = MagicMock()
    container.task_guard = MagicMock()
    container.results_lock = MagicMock()
    container.selector = MagicMock()
    container.tiered_router = MagicMock()
    container.adaptive_router = MagicMock()
    container.telemetry = MagicMock()
    container.policy_engine = MagicMock()
    container.project_planner = MagicMock()
    container.hook_registry = MagicMock()
    container.validator = MagicMock()
    container.decomposer = MagicMock()
    container.architect = MagicMock()
    container.executor = MagicMock()
    container.evaluator = MagicMock()
    container.generator = MagicMock()
    container.pipeline = MagicMock()
    container.event_bus = MagicMock()
    container.telemetry_store = MagicMock()
    container.semantic_cache = MagicMock()
    return container


@pytest.fixture
def orchestrator_mocked(mock_container):
    """Create Orchestrator with ServiceContainer.build() returning a mock."""
    with patch(
        "orchestrator.engine_core.container.ServiceContainer.build",
        return_value=mock_container,
    ):
        orch = Orchestrator(budget=Budget(max_usd=10.0, max_time_seconds=300))
        # Override _project_runner with a clean AsyncMock for assertion isolation
        orch._project_runner = AsyncMock()
        yield orch


class TestRunProjectCharacterization:
    """Characterization tests for Orchestrator.run_project() golden behavior."""

    @pytest.mark.asyncio
    async def test_run_project_returns_project_state(self, orchestrator_mocked):
        """run_project returns ProjectState with expected fields."""
        expected = ProjectState(
            project_description="Build a todo app",
            success_criteria="CRUD operations work",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected

        result = await orchestrator_mocked.run_project(
            project_description="Build a todo app",
            success_criteria="CRUD operations work",
        )

        assert isinstance(result, ProjectState)
        assert result.project_description == "Build a todo app"
        assert result.success_criteria == "CRUD operations work"

    @pytest.mark.asyncio
    async def test_run_project_delegates_to_project_runner(self, orchestrator_mocked):
        """run_project delegates all parameters to _project_runner.run_project()."""
        expected = ProjectState(
            project_description="Build API",
            success_criteria="All endpoints tested",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected

        await orchestrator_mocked.run_project(
            project_description="Build API",
            success_criteria="All endpoints tested",
            project_id="test-456",
            analyze_on_complete=True,
            output_dir=Path("/tmp/output"),
        )

        orchestrator_mocked._project_runner.run_project.assert_called_once_with(
            project_description="Build API",
            success_criteria="All endpoints tested",
            project_id="test-456",
            app_profile=None,
            analyze_on_complete=True,
            output_dir=Path("/tmp/output"),
        )

    @pytest.mark.asyncio
    async def test_run_project_with_defaults(self, orchestrator_mocked):
        """run_project works with minimal parameters (uses defaults)."""
        expected = ProjectState(
            project_description="Minimal project",
            success_criteria="It works",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected

        result = await orchestrator_mocked.run_project(
            project_description="Minimal project",
            success_criteria="It works",
        )

        assert result.project_description == "Minimal project"
        orchestrator_mocked._project_runner.run_project.assert_called_once()
        call_kwargs = orchestrator_mocked._project_runner.run_project.call_args[1]
        assert call_kwargs["app_profile"] is None
        assert not call_kwargs.get("analyze_on_complete", True)
        assert call_kwargs["output_dir"] is None

    @pytest.mark.asyncio
    async def test_run_project_signature_matches_public_contract(self, orchestrator_mocked):
        """Regression: run_project signature must not change during refactoring."""
        expected = ProjectState(
            project_description="Sig test",
            success_criteria="API unchanged",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected

        # Positional args
        result1 = await orchestrator_mocked.run_project(
            "Sig test", "API unchanged", "test-id"
        )
        assert result1 is not None

        # Keyword args
        result2 = await orchestrator_mocked.run_project(
            project_description="Sig test",
            success_criteria="API unchanged",
            project_id="test-id",
        )
        assert result2 is not None

        # Partial kwargs
        result3 = await orchestrator_mocked.run_project(
            "Sig test",
            "API unchanged",
            output_dir=Path("/tmp"),
        )
        assert result3 is not None
