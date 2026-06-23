"""
Unit tests for EvaluateStage.
Covers: exception handling, score normalization, evaluation_failed flags.
"""

import pytest

from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.engine_core.stages.evaluate import EvaluateStage
from orchestrator.models import Task, TaskType


@pytest.fixture
def mock_evaluator():
    """Factory that returns an evaluator with configurable behavior."""

    class _MockEvaluator:
        def __init__(self, raise_on_call=False, score=0.85, critique_text=""):
            self._raise = raise_on_call
            self._score = score
            self._critique = critique_text
            self.call_count = 0

        class _CritiqueReport:
            def __init__(self, score, critique_text):
                self.score = score
                self._critique = critique_text

            def to_prompt_context(self, max_items=10):
                return self._critique

        async def evaluate(self, task, output):
            self.call_count += 1
            if self._raise:
                raise RuntimeError("Evaluator crashed")
            return self._CritiqueReport(self._score, self._critique)

    return _MockEvaluator


@pytest.fixture
def ctx():
    task = Task(id="test-1", type=TaskType.CODE_GEN, prompt="test")
    return PipelineContext(task=task, output="some generated code")


@pytest.mark.unit
async def test_evaluate_happy_path(mock_evaluator, ctx):
    """Happy path: evaluator returns score=0.85, critique is injected."""
    evaluator = mock_evaluator(score=0.85, critique_text="looks good")
    stage = EvaluateStage(evaluator=evaluator)
    result = await stage.process(ctx)

    assert result.score == 0.85
    assert result.critique == "looks good"
    assert result.evaluation_failed is False
    assert result.evaluation_error == ""
    assert evaluator.call_count == 1


@pytest.mark.unit
async def test_evaluate_low_score(mock_evaluator, ctx):
    """Low score (e.g. 0.05) is preserved — not treated as failure."""
    evaluator = mock_evaluator(score=0.05, critique_text="poor quality")
    stage = EvaluateStage(evaluator=evaluator)
    result = await stage.process(ctx)

    assert result.score == 0.05
    assert result.evaluation_failed is False


@pytest.mark.unit
async def test_evaluate_exception_sets_failure_flags(mock_evaluator, ctx):
    """Exception in evaluator must set evaluation_failed, score=0.0."""
    evaluator = mock_evaluator(raise_on_call=True)
    stage = EvaluateStage(evaluator=evaluator)
    result = await stage.process(ctx)

    assert result.evaluation_failed is True
    assert "Evaluator crashed" in result.evaluation_error
    assert result.score == 0.0
    assert evaluator.call_count == 1


@pytest.mark.unit
async def test_evaluate_skip_empty_output(mock_evaluator):
    """If ctx.output is empty, evaluator is NOT called."""
    task = Task(id="test-2", type=TaskType.CODE_GEN, prompt="test")
    ctx_empty = PipelineContext(task=task, output="")
    evaluator = mock_evaluator()
    stage = EvaluateStage(evaluator=evaluator)
    result = await stage.process(ctx_empty)

    assert result.score == 0.0  # default
    assert evaluator.call_count == 0


@pytest.mark.unit
async def test_evaluate_exception_type_preserved(mock_evaluator, ctx):
    """Different exception types are captured in evaluation_error."""

    class _RaisingEvaluator:
        async def evaluate(self, task, output):
            raise ValueError("invalid output format")

    stage = EvaluateStage(evaluator=_RaisingEvaluator())
    result = await stage.process(ctx)

    assert result.evaluation_failed is True
    assert (
        "invalid output format" in result.evaluation_error
        or "ValueError" in result.evaluation_error
    )
