"""
Unit tests for GenerateStage — prompt construction and model selection.
"""

import pytest

pytestmark = pytest.mark.unit

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
        def __init__(self):
            self.spent_usd = 0.0
            self.remaining_usd = 1.0

        # Matches the real Budget.charge contract: async, (amount, phase).
        async def charge(self, amount, phase="generation"):
            self.spent_usd += amount
            return self.spent_usd

        # Matches the real Budget reservation contract used per call (B6).
        async def reserve(self, amount):
            return True

        async def commit_reservation(self, reserved_amount, actual_amount, phase="generation"):
            self.spent_usd += actual_amount

        async def release_reservation(self, amount):
            return None

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


@pytest.mark.unit
async def test_generate_skips_call_when_budget_exhausted(mock_client, mock_selector):
    """An exhausted budget must stop the call, not just be noticed afterwards."""
    from orchestrator.budget import Budget

    budget = Budget(max_usd=1.0)
    await budget.charge(1.0, "generation")
    client = mock_client()
    stage = GenerateStage(client=client, budget=budget, selector=mock_selector)
    task = Task(id="test-3", type=TaskType.CODE_GEN, prompt="test")

    result = await stage.process(PipelineContext(task=task))

    assert client.call_count == 0
    assert result.should_abort
    assert result.abort_reason == "budget_exhausted"
