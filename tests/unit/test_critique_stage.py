"""
Unit tests for CritiqueStage — code review and critique injection.
"""

import pytest

from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.engine_core.stages.critique import CritiqueStage
from orchestrator.models import Task, TaskType


@pytest.fixture
def mock_reviewer():
    """Factory for a mock reviewer that returns configurable critique."""

    class _MockResponse:
        def __init__(self, text):
            self.text = text
            self.cost_usd = 0.0
            self.usage = None

    class _MockReviewer:
        def __init__(self, critique="looks good", should_fail=False):
            self._critique = critique
            self._should_fail = should_fail
            self.call_count = 0

        async def call(self, model, prompt, system, max_tokens, **kw):
            self.call_count += 1
            self.last_prompt = prompt
            self.last_model = model
            if self._should_fail:
                raise RuntimeError("Reviewer crashed")
            return _MockResponse(self._critique)

    return _MockReviewer


@pytest.mark.unit
async def test_critique_injects_critique(mock_reviewer):
    """Critique output is injected into ctx.critique."""
    reviewer = mock_reviewer(critique="missing error handling")

    # get_reviewer_fn that always returns a different model than ctx.model
    def _get_reviewer(generator, task_type):
        return "reviewer-model"

    stage = CritiqueStage(client=reviewer, get_reviewer_fn=_get_reviewer)
    task = Task(id="test-1", type=TaskType.CODE_GEN, prompt="write api")
    ctx = PipelineContext(task=task, output="def api(): pass", model="gen-model")

    result = await stage.process(ctx)

    assert "missing error handling" in result.critique
    assert reviewer.call_count == 1


@pytest.mark.unit
async def test_critique_skips_empty_output(mock_reviewer):
    """If ctx.output is empty, reviewer is NOT called."""
    reviewer = mock_reviewer()
    task = Task(id="test-2", type=TaskType.CODE_GEN, prompt="test")
    ctx = PipelineContext(task=task, output="")

    stage = CritiqueStage(client=reviewer)
    result = await stage.process(ctx)

    assert result.critique == ""
    assert reviewer.call_count == 0


@pytest.mark.unit
async def test_critique_exception_does_not_crash(mock_reviewer):
    """Exception in reviewer is caught and logged, pipeline continues."""
    reviewer = mock_reviewer(should_fail=True)
    stage = CritiqueStage(client=reviewer)
    task = Task(id="test-3", type=TaskType.CODE_GEN, prompt="test")
    ctx = PipelineContext(task=task, output="code", model="mock-model")

    result = await stage.process(ctx)

    # Pipeline continues even when reviewer crashes
    assert result.critique == ""
    assert result.should_abort is False
