"""Characterization tests for Orchestrator.run_project() delegation contract."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.application.validators import ValidationError
from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.models import ProjectState, Budget as BudgetModel
from orchestrator.policy import JobSpec, PolicySet


@pytest.fixture
def mock_container():
    from unittest.mock import MagicMock

    c = MagicMock()
    c.client = MagicMock()
    c.client.is_available.return_value = True
    c.budget = Budget(max_usd=10.0, max_time_seconds=300)
    for attr in [
        "cache",
        "state_mgr",
        "pipeline_runner",
        "task_guard",
        "results_lock",
        "selector",
        "tiered_router",
        "adaptive_router",
        "telemetry",
        "policy_engine",
        "project_planner",
        "hook_registry",
        "validator",
        "decomposer",
        "architect",
        "executor",
        "evaluator",
        "generator",
        "pipeline",
        "event_bus",
        "telemetry_store",
        "semantic_cache",
    ]:
        setattr(c, attr, MagicMock())
    return c


@pytest.fixture
def orchestrator_mocked(mock_container):
    with patch(
        "orchestrator.engine_core.container.ServiceContainer.build", return_value=mock_container
    ):
        orch = Orchestrator(budget=Budget(max_usd=10.0, max_time_seconds=300))
        orch._project_runner = AsyncMock()
        yield orch


class TestRunProjectCharacterization:

    @pytest.mark.asyncio
    async def test_run_project_returns_project_state(self, orchestrator_mocked):
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

    @pytest.mark.asyncio
    async def test_run_project_delegates_to_project_runner(self, orchestrator_mocked):
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
        expected = ProjectState(
            project_description="Minimal",
            success_criteria="Works",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected
        result = await orchestrator_mocked.run_project(
            project_description="Minimal",
            success_criteria="Works",
        )
        assert result.project_description == "Minimal"
        call_kwargs = orchestrator_mocked._project_runner.run_project.call_args[1]
        assert call_kwargs["app_profile"] is None
        assert call_kwargs["output_dir"] is None

    @pytest.mark.asyncio
    async def test_run_project_signature_stable(self, orchestrator_mocked):
        expected = ProjectState(
            project_description="Sig test",
            success_criteria="API unchanged",
            budget=BudgetModel(max_usd=10.0, max_time_seconds=300),
        )
        orchestrator_mocked._project_runner.run_project.return_value = expected
        r1 = await orchestrator_mocked.run_project("Sig test", "API unchanged", "test-id")
        r2 = await orchestrator_mocked.run_project(
            project_description="Sig test", success_criteria="API unchanged", project_id="test-id"
        )
        r3 = await orchestrator_mocked.run_project(
            "Sig test", "API unchanged", output_dir=Path("/tmp")
        )
        assert r1 is not None and r2 is not None and r3 is not None


class TestRunProjectValidation:
    @pytest.mark.asyncio
    async def test_empty_project_description_raises(self, orchestrator_mocked):
        with pytest.raises(ValidationError, match="project_description"):
            await orchestrator_mocked.run_project("", "Works")

    @pytest.mark.asyncio
    async def test_oversized_project_description_raises(self, orchestrator_mocked):
        huge = "x" * 10_001
        with pytest.raises(ValidationError, match="project_description exceeds"):
            await orchestrator_mocked.run_project(huge, "Works")

    @pytest.mark.asyncio
    async def test_invalid_output_dir_raises(self, orchestrator_mocked):
        with pytest.raises(ValidationError, match="output_dir must be a Path"):
            await orchestrator_mocked.run_project("Valid", "Works", output_dir="/tmp/output")

    @pytest.mark.asyncio
    async def test_validation_prevents_runner_call(self, orchestrator_mocked):
        with pytest.raises(ValidationError):
            await orchestrator_mocked.run_project("", "Works")
        orchestrator_mocked._project_runner.run_project.assert_not_called()


class TestRunJobValidation:
    @pytest.mark.asyncio
    async def test_invalid_job_spec_raises(self, orchestrator_mocked):
        spec = JobSpec(
            project_description="Valid",
            success_criteria="Works",
            budget=Budget(max_usd=10.0, max_time_seconds=300),
            policy_set=PolicySet(),
            max_parallel_tasks=101,
        )
        with pytest.raises(ValidationError, match="max_parallel_tasks"):
            await orchestrator_mocked.run_job(spec)

    @pytest.mark.asyncio
    async def test_none_job_spec_raises(self, orchestrator_mocked):
        with pytest.raises(ValidationError, match="JobSpec must not be None"):
            await orchestrator_mocked.run_job(None)
