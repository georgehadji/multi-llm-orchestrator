"""
Unit tests for PipelineContext dataclass and its invariants.
"""

import pytest

from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.models import Task, TaskResult, TaskStatus, TaskType


@pytest.fixture
def sample_task():
    return Task(id="test-1", type=TaskType.CODE_GEN, prompt="build a widget")


@pytest.mark.unit
def test_default_fields(sample_task):
    """All fields have sensible defaults."""
    ctx = PipelineContext(task=sample_task)

    assert ctx.attempt == 0
    assert ctx.output == ""
    assert ctx.score == 0.0
    assert ctx.evaluation_failed is False
    assert ctx.evaluation_error == ""
    assert ctx.should_abort is False
    assert ctx.tokens_used == {"input": 0, "output": 0}
    assert ctx.cost_usd == 0.0
    assert ctx.skill_prefix == ""
    assert ctx.design_score == 0.0
    assert ctx.design_critique == ""


@pytest.mark.unit
def test_reset_for_retry(sample_task):
    """reset_for_retry clears abort flags but preserves task context."""
    ctx = PipelineContext(
        task=sample_task, score=0.5, should_abort=True, abort_reason="low quality"
    )
    ctx.reset_for_retry()

    assert ctx.should_abort is False
    assert ctx.abort_reason == ""
    # Score and other data are preserved (not reset)
    assert ctx.score == 0.5


@pytest.mark.unit
def test_to_task_result_defaults(sample_task):
    """to_task_result produces correct TaskResult from context."""
    ctx = PipelineContext(
        task=sample_task,
        output="print('hello')",
        score=0.92,
        model=None,  # Model.GPT_4O_MINI is the default
        tokens_used={"input": 100, "output": 200},
        cost_usd=0.05,
    )
    result: TaskResult = ctx.to_task_result(TaskStatus.COMPLETED)

    assert result.task_id == "test-1"
    assert result.output == "print('hello')"
    assert result.score == 0.92
    assert result.status == TaskStatus.COMPLETED
    assert result.tokens_used["input"] == 100
    assert result.cost_usd == 0.05


@pytest.mark.unit
def test_to_task_result_evaluation_failed(sample_task):
    """evaluation_failed is NOT yet propagated to TaskResult."""
    ctx = PipelineContext(
        task=sample_task,
        output="print('hello')",
        score=0.0,
        evaluation_failed=True,
        evaluation_error="crash",
    )
    result: TaskResult = ctx.to_task_result(TaskStatus.DEGRADED)

    assert result.status == TaskStatus.DEGRADED
    # NOTE: evaluation_failed is NOT in TaskResult currently.
    # This test documents the gap.


@pytest.mark.unit
def test_attempt_history_defaults(sample_task):
    """attempt_history starts empty; to_task_result preserves it."""
    ctx = PipelineContext(task=sample_task)
    assert ctx.attempt_history == []

    result = ctx.to_task_result(TaskStatus.DEGRADED)
    assert result.attempt_history == []
