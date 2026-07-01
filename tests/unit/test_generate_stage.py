"""
Unit tests for GenerateStage — prompt construction and model selection.
"""

import pytest

from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.engine_core.stages.generate import GenerateStage
from orchestrator.models import Task, TaskType


@pytest.fixture
def mock_client():
    """Factory for a mock LLMClient that returns configurable output."""

    class _MockResponse:
        def __init__(self, text):
            self.text = text
            self.cost_usd = 0.0
            self.usage = None

    class _MockClient:
        def __init__(self, output="result", should_fail=False):
            self._output = output
            self._should_fail = should_fail
            self.call_count = 0
            self.last_prompt = ""

        async def call(self, model, prompt, system, max_tokens, **kw):
            self.call_count += 1
            self.last_prompt = prompt
            self.last_system = system
            if self._should_fail:
                raise RuntimeError("API failure")
            return _MockResponse(self._output)

    return _MockClient


@pytest.fixture
def mock_selector():
    """Returns a fixed model for all task types."""

    class _MockSelector:
        def select(self, task_type):
            return "mock-model"

        def reviewer(self, generator, task_type):
            return "mock-reviewer"

    return _MockSelector()


@pytest.fixture
def mock_budget():
    class _MockBudget:
        def charge(self, model, tokens):
            return 0.0

        async def can_charge(self, model, tokens):
            return True

        async def charge_tokens(self, model, input_tokens, output_tokens):
            return 0.0

    return _MockBudget()


@pytest.mark.unit
async def test_generate_happy_path(mock_client, mock_selector, mock_budget):
    """Happy path: generate produces output, model is selected."""
    client = mock_client(output="def foo(): pass")
    stage = GenerateStage(client=client, budget=mock_budget, selector=mock_selector)
    task = Task(id="test-1", type=TaskType.CODE_GEN, prompt="write a function")
    ctx = PipelineContext(task=task)

    result = await stage.process(ctx)

    assert result.output == "def foo(): pass"
    assert result.model == "mock-model"
    assert client.call_count == 1


@pytest.mark.unit
async def test_generate_uses_skill_prefix(mock_client, mock_selector, mock_budget):
    """When skill_prefix is set, it's prepended to the system prompt."""
    client = mock_client(output="result")
    stage = GenerateStage(client=client, budget=mock_budget, selector=mock_selector)
    task = Task(id="test-2", type=TaskType.CODE_GEN, prompt="test")
    ctx = PipelineContext(task=task, skill_prefix="<skill>Do X</skill>")

    result = await stage.process(ctx)

    assert "<skill>Do X</skill>" in client.last_system
    assert result.output == "result"
