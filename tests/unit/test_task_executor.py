"""Tests for TaskExecutor — execution dispatch, context building, result construction."""

import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from orchestrator.application.task_executor import TaskExecutor, ExecutionContext
from orchestrator.application.critique_cycle import CritiqueState
from orchestrator.models import Task, TaskType, TaskResult, TaskStatus, Model

# ── Mock task_handlers to avoid import-time error ──────────────────────────


@pytest.fixture(autouse=True, scope="session")
def _mock_task_handlers():
    mock_mod = MagicMock()
    mock_mod.get_handler = MagicMock()
    sys.modules["orchestrator.task_handlers"] = mock_mod
    yield
    sys.modules.pop("orchestrator.task_handlers", None)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_cache():
    return MagicMock()


@pytest.fixture
def mock_semantic_cache():
    m = MagicMock()
    m.get_cached_pattern = MagicMock(return_value=None)
    return m


@pytest.fixture
def mock_cache_optimizer():
    m = MagicMock()
    m.get = AsyncMock(return_value=None)
    return m


@pytest.fixture
def mock_optim_config():
    return MagicMock()


@pytest.fixture
def mock_critique_cycle():
    return MagicMock()


@pytest.fixture
def mock_fallback_handler():
    return MagicMock()


@pytest.fixture
def mock_budget_enforcer():
    return MagicMock()


@pytest.fixture
def mock_dep_resolver():
    return MagicMock()


@pytest.fixture
def executor(
    mock_client,
    mock_cache,
    mock_semantic_cache,
    mock_cache_optimizer,
    mock_optim_config,
    mock_critique_cycle,
    mock_fallback_handler,
    mock_budget_enforcer,
    mock_dep_resolver,
):
    return TaskExecutor(
        client=mock_client,
        cache=mock_cache,
        semantic_cache=mock_semantic_cache,
        cache_optimizer=mock_cache_optimizer,
        optim_config=mock_optim_config,
        critique_cycle=mock_critique_cycle,
        fallback_handler=mock_fallback_handler,
        budget_enforcer=mock_budget_enforcer,
        dependency_resolver=mock_dep_resolver,
    )


@pytest.fixture
def sample_task():
    return Task(
        id="task-1",
        type=TaskType.CODE_GEN,
        prompt="Write a function",
        target_path="src/main.py",
        max_output_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════════════════════
# _build_cached_result
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCachedResult:
    def test_returns_result_with_cached_data(self, executor, sample_task):
        ctx = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            cached_result={
                "response": "def f(): pass",
                "tokens_input": 50,
                "tokens_output": 100,
                "cost": 0.001,
            },
        )
        r = executor._build_cached_result(sample_task, ctx)
        assert r.output == "def f(): pass"
        assert r.status == TaskStatus.COMPLETED
        assert r.iterations == 0


# ═══════════════════════════════════════════════════════════════════════════
# _build_result_from_cycle
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildResultFromCycle:
    def test_high_score_is_completed(self, executor, sample_task):
        s = CritiqueState(best_output="great", best_score=0.95, total_cost=0.05)
        ctx = ExecutionContext(
            task=sample_task, primary_model=Model.GPT_4O_MINI, reviewer_model=None
        )
        r = executor._build_result_from_cycle(sample_task, s, ctx)
        assert r.status == TaskStatus.COMPLETED

    def test_medium_score_is_degraded(self, executor, sample_task):
        s = CritiqueState(best_output="ok", best_score=0.78)
        ctx = ExecutionContext(
            task=sample_task, primary_model=Model.GPT_4O_MINI, reviewer_model=None
        )
        r = executor._build_result_from_cycle(sample_task, s, ctx)
        assert r.status == TaskStatus.DEGRADED

    def test_low_score_is_failed(self, executor, sample_task):
        s = CritiqueState(best_output="bad", best_score=0.45)
        ctx = ExecutionContext(
            task=sample_task, primary_model=Model.GPT_4O_MINI, reviewer_model=None
        )
        r = executor._build_result_from_cycle(sample_task, s, ctx)
        assert r.status == TaskStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════
# _build_full_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildFullPrompt:
    def test_without_context(self, executor, sample_task):
        ctx = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="",
        )
        assert executor._build_full_prompt(sample_task, ctx) == sample_task.prompt

    def test_with_context_appends(self, executor, sample_task):
        ctx = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="prev code",
        )
        assert "prev code" in executor._build_full_prompt(sample_task, ctx)

    def test_code_review_label(self, executor):
        t = Task(id="r1", type=TaskType.CODE_REVIEW, prompt="Review this")
        ctx = ExecutionContext(
            task=t,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="src code",
        )
        assert "SOURCE CODE TO REVIEW" in executor._build_full_prompt(t, ctx)


# ═══════════════════════════════════════════════════════════════════════════
# _build_failure_result
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildFailureResult:
    def test_failure_structure(self, executor, sample_task):
        r = executor._build_failure_result(sample_task, "No models")
        assert r.status == TaskStatus.FAILED
        assert r.score == 0.0
        assert r.critique == "No models"


# ═══════════════════════════════════════════════════════════════════════════
# execute_task — full dispatch flow
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_cached_result_short_circuits(self, executor, sample_task):
        executor.dependency_resolver.get_dependency_context.return_value = ""
        executor.cache_optimizer.get = AsyncMock(
            return_value={"response": "cached", "tokens_input": 5, "tokens_output": 10, "cost": 0.0}
        )
        r = await executor.execute_task(sample_task, {}, {})
        assert r.output == "cached"
        assert r.score == 0.85
        assert r.iterations == 0

    @pytest.mark.asyncio
    async def test_handler_key_error_falls_through(self, executor, sample_task):
        executor.cache_optimizer.get.return_value = None
        executor.dependency_resolver.get_dependency_context.return_value = ""
        executor.fallback_handler.get_available_models.return_value = [Model.GPT_4O_MINI]
        executor.fallback_handler.select_reviewer.return_value = Model.GPT_4O
        executor.critique_cycle.run_cycle = AsyncMock(
            return_value=CritiqueState(best_output="ok", best_score=0.82, total_cost=0.01)
        )
        import sys

        sys.modules["orchestrator.task_handlers"].get_handler.side_effect = KeyError("no handler")
        r = await executor.execute_task(sample_task, {}, {})
        assert r.score == 0.82
        executor.critique_cycle.run_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_failure_when_no_models(self, executor, sample_task):
        executor.cache_optimizer.get.return_value = None
        executor.dependency_resolver.get_dependency_context.return_value = ""
        executor.fallback_handler.get_available_models.return_value = []
        r = await executor.execute_task(sample_task, {}, {})
        assert r.status == TaskStatus.FAILED
        assert "No models" in r.critique

    @pytest.mark.asyncio
    async def test_typed_handler_succeeds(self, executor, sample_task):
        import sys

        # Ensure get_handler doesn't have side_effect from previous test
        sys.modules["orchestrator.task_handlers"].get_handler.side_effect = None
        handler = MagicMock()
        handler.execute = AsyncMock(
            return_value=MagicMock(
                output="handler output",
                task_id="task-1",
                status=TaskStatus.COMPLETED,
                score=0.9,
                tokens_used={"input": 0, "output": 0},
                iterations=1,
                cost_usd=0.0,
                critique="",
                task_type=TaskType.CODE_GEN.value,
            )
        )
        sys.modules["orchestrator.task_handlers"].get_handler.return_value = lambda: handler
        executor.cache_optimizer.get.return_value = None
        executor.dependency_resolver.get_dependency_context.return_value = ""
        executor.critique_cycle.run_cycle = AsyncMock()
        r = await executor.execute_task(sample_task, {}, {})
        assert r is not None
        handler.execute.assert_called_once()
