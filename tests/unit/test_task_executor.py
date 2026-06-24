"""
Unit tests for TaskExecutor — individual task execution logic.

Tests cover:
- Result construction from cached context, critique cycle state, and failures
- Full prompt building with dependency context
- Execution context building
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from orchestrator.application.task_executor import TaskExecutor, ExecutionContext
from orchestrator.application.critique_cycle import CritiqueState
from orchestrator.models import Task, TaskType, TaskResult, TaskStatus, Model


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_cache():
    return MagicMock()


@pytest.fixture
def mock_semantic_cache():
    return MagicMock()


@pytest.fixture
def mock_cache_optimizer():
    return MagicMock()


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


# ═══════════════════════════════════════════════════════════════════════════════
# _build_cached_result
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildCachedResult:
    """TaskExecutor._build_cached_result() converts cache hit to TaskResult."""

    def test_returns_task_result_with_cached_data(self, executor, sample_task):
        """Cache hit produces a COMPLETED TaskResult with cached response."""
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            cached_result={
                "response": "def foo():\n    pass",
                "tokens_input": 50,
                "tokens_output": 100,
                "cost": 0.001,
            },
        )
        result = executor._build_cached_result(sample_task, context)
        assert isinstance(result, TaskResult)
        assert result.output == "def foo():\n    pass"
        assert result.status == TaskStatus.COMPLETED
        assert result.score == 0.85

    def test_cached_result_zero_iterations(self, executor, sample_task):
        """Cached results report 0 iterations."""
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            cached_result={"response": "ok", "tokens_input": 0, "tokens_output": 0},
        )
        result = executor._build_cached_result(sample_task, context)
        assert result.iterations == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _build_result_from_cycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildResultFromCycle:
    """TaskExecutor._build_result_from_cycle() maps CritiqueState to TaskResult."""

    def test_high_score_is_completed(self, executor, sample_task):
        """Score >= 0.9 maps to COMPLETED status."""
        state = CritiqueState(best_output="great code", best_score=0.95, total_cost=0.05)
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=Model.GPT_4O,
        )
        result = executor._build_result_from_cycle(sample_task, state, context)
        assert result.status == TaskStatus.COMPLETED
        assert result.output == "great code"
        assert result.score == 0.95

    def test_medium_score_is_degraded(self, executor, sample_task):
        """0.7 <= score < 0.9 maps to DEGRADED status."""
        state = CritiqueState(best_output="ok code", best_score=0.78)
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
        )
        result = executor._build_result_from_cycle(sample_task, state, context)
        assert result.status == TaskStatus.DEGRADED

    def test_low_score_is_failed(self, executor, sample_task):
        """Score < 0.7 maps to FAILED status."""
        state = CritiqueState(best_output="bad code", best_score=0.45)
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
        )
        result = executor._build_result_from_cycle(sample_task, state, context)
        assert result.status == TaskStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# _build_failure_result
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildFailureResult:
    """TaskExecutor._build_failure_result() constructs error results."""

    def test_failure_result_structure(self, executor, sample_task):
        """Failure result has empty output and provided reason as critique."""
        result = executor._build_failure_result(sample_task, "No models available")
        assert result.status == TaskStatus.FAILED
        assert result.output == ""
        assert result.score == 0.0
        assert result.critique == "No models available"
        assert result.cost_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _build_full_prompt
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildFullPrompt:
    """TaskExecutor._build_full_prompt() incorporates dependency context."""

    def test_without_dep_context_prompt_is_task_prompt(self, executor, sample_task):
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="",
        )
        prompt = executor._build_full_prompt(sample_task, context)
        assert prompt == sample_task.prompt

    def test_with_dep_context_appends(self, executor, sample_task):
        context = ExecutionContext(
            task=sample_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="Previous function: add()",
        )
        prompt = executor._build_full_prompt(sample_task, context)
        assert "Previous function: add()" in prompt

    def test_code_review_gets_explicit_label(self, executor, sample_task):
        """CODE_REVIEW tasks get a 'SOURCE CODE TO REVIEW' header."""
        review_task = Task(id="r1", type=TaskType.CODE_REVIEW, prompt="Review this")
        context = ExecutionContext(
            task=review_task,
            primary_model=Model.GPT_4O_MINI,
            reviewer_model=None,
            dependency_context="source code",
        )
        prompt = executor._build_full_prompt(review_task, context)
        assert "SOURCE CODE TO REVIEW" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# _build_execution_context
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildExecutionContext:
    """TaskExecutor._build_execution_context() resolves dependencies and cache."""

    @pytest.mark.asyncio
    async def test_gathers_dependency_context(self, executor, sample_task):
        """Calls dependency_resolver.get_dependency_context()."""
        tasks = {sample_task.id: sample_task}
        results = {}
        executor.dependency_resolver.get_dependency_context.return_value = "context from deps"

        context = await executor._build_execution_context(sample_task, tasks, results)

        assert context.dependency_context == "context from deps"

    @pytest.mark.asyncio
    async def test_resolves_cache_path(self, executor, sample_task):
        """With dependencies, cache optimizer is NOT queried."""
        tasks = {sample_task.id: sample_task}
        results = {}
        executor.dependency_resolver.get_dependency_context.return_value = "has deps"
        executor.cache_optimizer.get = AsyncMock()

        context = await executor._build_execution_context(sample_task, tasks, results)

        # Cached result should be None because cache_optimizer.get wasn't called
        # (the code skips cache when dependency_context is truthy)
        assert context.cached_result is None
        executor.cache_optimizer.get.assert_not_called()
