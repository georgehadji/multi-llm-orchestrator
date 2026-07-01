"""Unit tests for CritiqueCycle — score extraction, code cleaning, model params."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from orchestrator.application.critique_cycle import CritiqueCycle, CritiqueState
from orchestrator.models import Task, TaskType, Model


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def cycle(mock_client):
    return CritiqueCycle(client=mock_client, max_iterations=5)


@pytest.fixture
def sample_code_task():
    return Task(
        id="test-1",
        type=TaskType.CODE_GEN,
        prompt="Write hello",
        target_path="src/main.py",
        max_output_tokens=4096,
    )


# ── Score extraction ──────────────────────────────────────────────────────────


class TestExtractScore:
    def test_json_score(self, cycle):
        assert cycle._extract_score('{"score": 0.85}') == 0.85

    def test_json_fenced(self, cycle):
        assert cycle._extract_score('```json\n{"score": 0.77}\n```') == 0.77

    def test_inline_score(self, cycle):
        assert cycle._extract_score("Score: 0.75") == 0.75

    def test_clamps_negative(self, cycle):
        assert cycle._extract_score('{"score": -0.5}') == 0.0

    def test_clamps_above_one(self, cycle):
        assert cycle._extract_score('{"score": 1.5}') == 1.0

    def test_default_on_no_match(self, cycle):
        assert cycle._extract_score("Looks good.") == 0.5


# ── Code output cleaning ──────────────────────────────────────────────────────


class TestCleanCodeOutput:
    def test_removes_fences(self, cycle):
        assert cycle._clean_code_output("```python\npass\n```") == "pass"

    def test_removes_placeholders(self, cycle):
        cleaned = cycle._clean_code_output("def f():\n    // Add your code here\n    pass")
        assert "Add your code" not in cleaned

    def test_preserves_valid_code(self, cycle):
        assert cycle._clean_code_output("def f(): pass") == "def f(): pass"


# ── Model parameters ──────────────────────────────────────────────────────────


class TestGetModelParams:
    def test_reasoning_model_gets_240s(self, cycle):
        t, m = cycle._get_model_params(Model.GPT_5, TaskType.CODE_GEN, 4096)
        assert t == 240

    def test_code_gen_gets_120s(self, cycle):
        t, _ = cycle._get_model_params(Model.GPT_4O_MINI, TaskType.CODE_GEN, 4096)
        assert t == 120

    def test_other_types_get_60s(self, cycle):
        t, _ = cycle._get_model_params(Model.GPT_4O_MINI, TaskType.WRITING, 4096)
        assert t == 60


# ── Language detection ────────────────────────────────────────────────────────


class TestDetectLanguage:
    def test_py_extension(self, cycle):
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="x", target_path="app.py")
        assert cycle._detect_language(task) == "python"

    def test_ts_extension(self, cycle):
        task = Task(id="t2", type=TaskType.CODE_GEN, prompt="x", target_path="index.ts")
        assert cycle._detect_language(task) == "typescript"

    def test_defaults_to_python(self, cycle):
        task = Task(id="t3", type=TaskType.CODE_GEN, prompt="x")
        assert cycle._detect_language(task) == "python"


# ── Syntax validation ────────────────────────────────────────────────────────


class TestValidateSyntax:
    def test_valid(self, cycle):
        assert cycle._validate_syntax("x = 1") is True

    def test_invalid(self, cycle):
        assert cycle._validate_syntax("x = ") is False


# ── Function name extraction ─────────────────────────────────────────────────


class TestExtractFunctionName:
    def test_function(self, cycle):
        assert cycle._extract_function_name("def hello(): pass") == "hello"

    def test_class_name(self, cycle):
        assert cycle._extract_function_name("class Foo: pass") == "Foo"

    def test_ignores_dunder(self, cycle):
        code = "class M:\n    def __init__(self): pass\n    def run(self): pass"
        assert cycle._extract_function_name(code) == "run"


# ── Cycle iteration ──────────────────────────────────────────────────────────


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_excellence_early_exit(self, cycle, sample_code_task):
        with patch.object(cycle, "_generate") as gen, patch.object(cycle, "_critique") as crit:
            gen.return_value = MagicMock(
                text="perfect", cost_usd=0.01, input_tokens=10, output_tokens=5
            )
            crit.return_value = MagicMock(
                text='{"score": 0.97}', cost_usd=0.005, input_tokens=5, output_tokens=3
            )
            state = await cycle.run_cycle(
                sample_code_task, Model.GPT_4O_MINI, Model.GPT_4O, "Write"
            )
            assert len(state.scores_history) == 1

    @pytest.mark.asyncio
    async def test_generation_failure_returns_early(self, cycle, sample_code_task):
        with patch.object(cycle, "_generate") as gen:
            gen.return_value = None
            state = await cycle.run_cycle(
                sample_code_task, Model.GPT_4O_MINI, Model.GPT_4O, "Write"
            )
            assert state.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_no_reviewer_skips_critique(self, cycle, sample_code_task):
        with patch.object(cycle, "_generate") as gen, patch.object(cycle, "_critique") as crit:
            gen.return_value = MagicMock(
                text="output", cost_usd=0.01, input_tokens=10, output_tokens=5
            )
            await cycle.run_cycle(sample_code_task, Model.GPT_4O_MINI, None, "Write")
            crit.assert_not_called()
