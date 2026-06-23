"""
Unit tests for PipelineExecutor — task execution loop.
Tests the execute() method with mocked pipeline and selector.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.engine_core.pipeline_executor import PipelineExecutor
from orchestrator.models import Task, TaskResult, TaskStatus, TaskType


@pytest.fixture
def sample_task():
    return Task(
        id="test-1",
        type=TaskType.CODE_GEN,
        prompt="build a widget",
        acceptance_threshold=0.7,
    )


@pytest.fixture
def mock_pipeline():
    """A mock TaskPipeline that simulates successful execution."""
    p = MagicMock()

    async def mock_run(ctx):
        ctx.output = "def hello(): pass"
        ctx.score = 0.85
        ctx.tokens_used = {"input": 100, "output": 50}
        ctx.cost_usd = 0.005
        ctx.should_abort = True
        return ctx

    p.run = AsyncMock(side_effect=mock_run)
    return p


@pytest.fixture
def mock_selector():
    s = MagicMock()
    s.select = MagicMock(return_value=None)  # None = use task.preferred_model
    return s


@pytest.fixture
def executor(mock_pipeline, mock_selector):
    return PipelineExecutor(
        pipeline=mock_pipeline,
        selector=mock_selector,
        background_tasks=set(),
    )


@pytest.mark.unit
async def test_execute_returns_task_result(executor, sample_task):
    """execute() returns a TaskResult with the expected fields."""
    result = await executor.execute(sample_task)

    assert isinstance(result, TaskResult)
    assert result.task_id == "test-1"
    assert result.score == 0.85
    assert result.tokens_used == {"input": 100, "output": 50}
    assert result.cost_usd == 0.005


@pytest.mark.unit
async def test_execute_completed_status(executor, sample_task):
    """Score above threshold → COMPLETED status."""
    result = await executor.execute(sample_task)
    assert result.status == TaskStatus.COMPLETED


@pytest.mark.unit
async def test_execute_degraded_status(executor, sample_task):
    """Score below acceptance_threshold → DEGRADED status."""
    p = MagicMock()

    async def low_score(ctx):
        ctx.output = "def hello(): pass"
        ctx.score = 0.3  # below 0.7 threshold
        ctx.tokens_used = {"input": 100, "output": 50}
        ctx.cost_usd = 0.005
        ctx.should_abort = True
        return ctx

    p.run = AsyncMock(side_effect=low_score)
    exec_low = PipelineExecutor(pipeline=p, selector=executor._selector, background_tasks=set())
    result = await exec_low.execute(sample_task)
    assert result.status == TaskStatus.DEGRADED


@pytest.mark.unit
async def test_execute_failed_on_stage_error(executor, sample_task):
    """abort_reason starting with 'stage_error' → FAILED status."""
    p = MagicMock()

    async def stage_error(ctx):
        ctx.output = ""
        ctx.score = 0.0
        ctx.should_abort = True
        ctx.abort_reason = "stage_error_critique_timeout"
        ctx.tokens_used = {"input": 10, "output": 0}
        ctx.cost_usd = 0.001
        return ctx

    p.run = AsyncMock(side_effect=stage_error)
    exec_err = PipelineExecutor(pipeline=p, selector=executor._selector, background_tasks=set())
    result = await exec_err.execute(sample_task)
    assert result.status == TaskStatus.FAILED


@pytest.mark.unit
async def test_execute_retry_loop(executor, sample_task):
    """execute() retries on 'retry_for_quality' abort_reason."""
    call_count = 0

    async def retry_then_succeed(ctx):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            ctx.should_abort = True
            ctx.abort_reason = "retry_for_quality"
        else:
            ctx.output = "final output"
            ctx.score = 0.9
            ctx.should_abort = True
            ctx.abort_reason = ""
        ctx.tokens_used = {"input": 50, "output": 25}
        ctx.cost_usd = 0.002
        return ctx

    p = MagicMock()
    p.run = AsyncMock(side_effect=retry_then_succeed)
    exec_retry = PipelineExecutor(
        pipeline=p, selector=executor._selector, background_tasks=set()
    )
    result = await exec_retry.execute(sample_task)
    assert result.score == 0.9
    assert result.status == TaskStatus.COMPLETED
    assert call_count == 3  # two retries + one final success


@pytest.mark.unit
async def test_execute_ara_retry(executor, sample_task):
    """execute() retries on 'ara_retry' abort_reason."""
    call_count = 0

    async def ara_then_succeed(ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            ctx.should_abort = True
            ctx.abort_reason = "ara_retry"
        else:
            ctx.output = "ara output"
            ctx.score = 0.95
            ctx.should_abort = True
            ctx.abort_reason = ""
        ctx.tokens_used = {"input": 50, "output": 25}
        ctx.cost_usd = 0.002
        return ctx

    p = MagicMock()
    p.run = AsyncMock(side_effect=ara_then_succeed)
    exec_ara = PipelineExecutor(
        pipeline=p, selector=executor._selector, background_tasks=set()
    )
    result = await exec_ara.execute(sample_task)
    assert result.score == 0.95
    assert call_count == 2


@pytest.mark.unit
async def test_execute_uses_task_preferred_model(executor, sample_task):
    """When task has preferred_model, selector is not consulted."""
    sample_task.preferred_model = "claude-3-opus"
    result = await executor.execute(sample_task)
    assert result is not None
    # selector.select should NOT have been called
    executor._selector.select.assert_not_called()


@pytest.mark.unit
async def test_execute_model_select_fallback(executor, sample_task):
    """When no preferred_model, selector.select is called."""
    p = MagicMock()

    async def run_with_model(ctx):
        ctx.output = "code"
        ctx.score = 0.8
        ctx.should_abort = True
        ctx.tokens_used = {"input": 50, "output": 25}
        ctx.cost_usd = 0.002
        return ctx

    p.run = AsyncMock(side_effect=run_with_model)
    selector = MagicMock()
    selector.select = MagicMock(return_value="gpt-4")

    exec_fallback = PipelineExecutor(pipeline=p, selector=selector, background_tasks=set())
    result = await exec_fallback.execute(sample_task)

    selector.select.assert_called_once_with(sample_task.type)


@pytest.mark.unit
async def test_to_task_result_has_tokens(executor, sample_task):
    """TaskResult should correctly reflect tokens_used."""
    result = await executor.execute(sample_task)
    assert result.tokens_used == {"input": 100, "output": 50}
