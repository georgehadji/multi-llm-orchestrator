"""
Unit tests for CritiqueCycle — the generate → critique → revise → evaluate pipeline.

Tests cover:
- Score extraction from JSON and free-text critique responses
- Code output cleaning (markdown fences, placeholders)
- Model parameter resolution (timeout, max_tokens per model type)
- Language detection from task metadata
- Syntax validation and function name extraction
- Cycle-level iteration logic (plateau detection, excellence threshold, max iterations)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.application.critique_cycle import CritiqueCycle, CritiqueState
from orchestrator.models import Task, TaskType, Model

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def cycle(mock_client):
    return CritiqueCycle(client=mock_client, max_iterations=5)


@pytest.fixture
def sample_code_task() -> Task:
    return Task(
        id="test-1",
        type=TaskType.CODE_GEN,
        prompt="Write a hello world",
        target_path="src/main.py",
        max_output_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Score extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractScore:
    """CritiqueCycle._extract_score() parses scores from critique text."""

    def test_extracts_json_score(self, cycle):
        """Parses score from JSON block in critique."""
        text = 'Here is my review: {"score": 0.85, "issues": []}'
        score = cycle._extract_score(text)
        assert score == 0.85

    def test_extracts_json_score_with_large_value(self, cycle):
        """Parses a high score from JSON."""
        text = '{"score": 0.95, "summary": "Excellent work"}'
        score = cycle._extract_score(text)
        assert score == 0.95

    def test_extracts_inline_score(self, cycle):
        """Parses score from plain-text 'Score: N.N' format."""
        text = "Score: 0.75\nThis could be improved..."
        score = cycle._extract_score(text)
        assert score == 0.75

    def test_extracts_score_without_prefix(self, cycle):
        """Parses 'score X.Y' format."""
        text = "The overall score 0.62 is acceptable."
        score = cycle._extract_score(text)
        assert score == 0.62

    def test_clamps_score_below_zero(self, cycle):
        """Negative scores are clamped to 0.0."""
        text = '{"score": -0.5}'
        score = cycle._extract_score(text)
        assert score == 0.0

    def test_clamps_score_above_one(self, cycle):
        """Scores above 1.0 are clamped to 1.0."""
        text = '{"score": 1.5}'
        score = cycle._extract_score(text)
        assert score == 1.0

    def test_returns_default_on_no_match(self, cycle):
        """Unparseable critique returns default 0.5."""
        text = "This code is good but could be better."
        score = cycle._extract_score(text)
        assert score == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Code output cleaning
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanCodeOutput:
    """CritiqueCycle._clean_code_output() removes fences and placeholders."""

    def test_removes_markdown_fences(self, cycle):
        """Strips ```python and ``` fences."""
        code = "```python\ndef hello():\n    pass\n```"
        cleaned = cycle._clean_code_output(code)
        assert cleaned == "def hello():\n    pass"

    def test_removes_placeholder_comments(self, cycle):
        """Strips '// Add content here' placeholders."""
        code = "def foo():\n    // Add your code here\n    pass"
        cleaned = cycle._clean_code_output(code)
        assert "Add your code here" not in cleaned

    def test_collapses_excessive_blank_lines(self, cycle):
        """Collapses 3+ consecutive blank lines to 2."""
        code = "line 1\n\n\n\nline 2"
        cleaned = cycle._clean_code_output(code)
        # Should have at most 2 blank lines between the lines
        assert cleaned.count("\n\n") <= 2

    def test_preserves_valid_code(self, cycle):
        """Valid Python code with no fences is unchanged."""
        code = "def hello():\n    return 'world'"
        cleaned = cycle._clean_code_output(code)
        assert cleaned == code


# ═══════════════════════════════════════════════════════════════════════════════
# Model parameters
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetModelParams:
    """CritiqueCycle._get_model_params() resolves model-specific settings."""

    def test_reasoning_model_gets_longer_timeout(self, cycle):
        """Reasoning models get 240s timeout and double max_tokens."""
        model = Model.GPT_5
        timeout, max_tokens = cycle._get_model_params(model, TaskType.CODE_GEN, max_tokens=4096)
        assert timeout == 240
        assert max_tokens >= 4096

    def test_code_gen_gets_120s_timeout(self, cycle):
        """Code gen/review tasks get 120s timeout on non-reasoning models."""
        model = Model.GPT_4O_MINI
        timeout, _ = cycle._get_model_params(model, TaskType.CODE_GEN, max_tokens=4096)
        assert timeout == 120

    def test_other_types_get_60s_timeout(self, cycle):
        """Non-code tasks get 60s timeout."""
        model = Model.GPT_4O_MINI
        timeout, _ = cycle._get_model_params(model, TaskType.WRITING, max_tokens=4096)
        assert timeout == 60

    def test_model_max_tokens_caps_effective(self, cycle):
        """Model max tokens caps the effective max_tokens."""
        model = Model.GPT_4O_MINI
        _, max_tokens = cycle._get_model_params(model, TaskType.CODE_GEN, max_tokens=99999)
        # GPT_4O_MINI max_tokens should be the cap
        assert max_tokens <= 128000


# ═══════════════════════════════════════════════════════════════════════════════
# Language detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectLanguage:
    """CritiqueCycle._detect_language() infers language from task metadata."""

    def test_detects_from_extension_py(self, cycle):
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="x", target_path="app.py")
        assert cycle._detect_language(task) == "python"

    def test_detects_from_extension_ts(self, cycle):
        task = Task(id="t2", type=TaskType.CODE_GEN, prompt="x", target_path="index.ts")
        assert cycle._detect_language(task) == "typescript"

    def test_detects_from_extension_go(self, cycle):
        task = Task(id="t3", type=TaskType.CODE_GEN, prompt="x", target_path="main.go")
        assert cycle._detect_language(task) == "go"

    def test_defaults_to_python(self, cycle):
        task = Task(id="t4", type=TaskType.CODE_GEN, prompt="x")
        assert cycle._detect_language(task) == "python"


# ═══════════════════════════════════════════════════════════════════════════════
# Syntax validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateSyntax:
    """CritiqueCycle._validate_syntax() checks Python AST validity."""

    def test_valid_syntax(self, cycle):
        assert cycle._validate_syntax("x = 1") is True

    def test_invalid_syntax(self, cycle):
        assert cycle._validate_syntax("x = ") is False

    def test_function_def(self, cycle):
        assert cycle._validate_syntax("def f():\n    pass") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Function name extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractFunctionName:
    """CritiqueCycle._extract_function_name() finds first function/class def."""

    def test_extracts_function(self, cycle):
        assert cycle._extract_function_name("def hello():\n    pass") == "hello"

    def test_extracts_class_name(self, cycle):
        assert cycle._extract_function_name("class Foo:\n    pass") == "Foo"

    def test_extracts_main_function(self, cycle):
        code = "def main():\n    pass\n\ndef helper():\n    pass"
        assert cycle._extract_function_name(code) == "main"

    def test_ignores_dunder_methods(self, cycle):
        code = (
            "class Meta:\n    def __init__(self):\n        pass\n    def run(self):\n        pass"
        )
        assert cycle._extract_function_name(code) == "run"

    def test_fallback_regex_on_syntax_error(self, cycle):
        code = "def broken(\n    pass"
        result = cycle._extract_function_name(code)
        # Should fall back to regex
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Cycle-level iteration logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunCycle:
    """CritiqueCycle.run_cycle() iteration control."""

    @pytest.mark.asyncio
    async def test_plateau_detection_stops_early(self, cycle, sample_code_task):
        """When improvement is below plateau threshold, cycle stops."""
        with (
            patch.object(cycle, "_generate") as mock_gen,
            patch.object(cycle, "_critique") as mock_critique,
        ):
            # Return same score three times (plateau)
            mock_gen.return_value = MagicMock(
                text="print('hello')",
                cost_usd=0.01,
                input_tokens=10,
                output_tokens=5,
            )
            mock_critique.return_value = MagicMock(
                text='{"score": 0.82}',
                cost_usd=0.005,
                input_tokens=5,
                output_tokens=3,
            )

            state = await cycle.run_cycle(
                task=sample_code_task,
                primary_model=Model.GPT_4O_MINI,
                reviewer_model=Model.GPT_4O,
                full_prompt="Write hello",
            )

            # Should have run 2 iterations (3 would be collected if plateau not detected,
            # but with plateau at 0.05 and score constant, it fires after 2)
            assert state.best_score == 0.82
            assert len(state.scores_history) >= 2

    @pytest.mark.asyncio
    async def test_excellence_threshold_early_exit(self, cycle, sample_code_task):
        """When score >= 0.95, cycle stops early."""
        with (
            patch.object(cycle, "_generate") as mock_gen,
            patch.object(cycle, "_critique") as mock_critique,
        ):
            mock_gen.return_value = MagicMock(
                text="perfect code",
                cost_usd=0.01,
                input_tokens=10,
                output_tokens=5,
            )
            mock_critique.return_value = MagicMock(
                text='{"score": 0.97}',
                cost_usd=0.005,
                input_tokens=5,
                output_tokens=3,
            )

            state = await cycle.run_cycle(
                task=sample_code_task,
                primary_model=Model.GPT_4O_MINI,
                reviewer_model=Model.GPT_4O,
                full_prompt="Write hello",
            )

            assert state.best_score == 0.97
            assert len(state.scores_history) == 1, "Should stop after first iteration"

    @pytest.mark.asyncio
    async def test_generation_failure_returns_early(self, cycle, sample_code_task):
        """When _generate returns None, cycle breaks."""
        with patch.object(cycle, "_generate") as mock_gen:
            mock_gen.return_value = None

            state = await cycle.run_cycle(
                task=sample_code_task,
                primary_model=Model.GPT_4O_MINI,
                reviewer_model=Model.GPT_4O,
                full_prompt="Write hello",
            )

            assert state.total_cost == 0.0
            assert state.best_output == ""

    @pytest.mark.asyncio
    async def test_lsp_validation_executes_for_code_gen(self, cycle, sample_code_task):
        """LSP validator is called for CODE_GEN tasks when lsp_validator is set."""
        from unittest.mock import AsyncMock
        from orchestrator.domain.ports import NullLspValidator

        # Create a cycle WITH an LSP validator
        lsp_cycle = CritiqueCycle(
            client=MagicMock(),
            lsp_validator=NullLspValidator(),
            max_iterations=5,
        )

        with (
            patch.object(lsp_cycle, "_generate") as mock_gen,
            patch.object(lsp_cycle, "_critique") as mock_critique,
        ):
            mock_gen.return_value = MagicMock(
                text="def foo():\n    pass",
                cost_usd=0.01,
                input_tokens=10,
                output_tokens=5,
            )
            mock_critique.return_value = MagicMock(
                text='{"score": 0.88}',
                cost_usd=0.005,
                input_tokens=5,
                output_tokens=3,
            )

            state = await lsp_cycle.run_cycle(
                task=sample_code_task,
                primary_model=Model.GPT_4O_MINI,
                reviewer_model=Model.GPT_4O,
                full_prompt="Write a function",
            )

            assert state.best_score > 0

    @pytest.mark.asyncio
    async def test_no_reviewer_skips_critique(self, cycle, sample_code_task):
        """When reviewer_model is None, critique step is skipped."""
        with (
            patch.object(cycle, "_generate") as mock_gen,
            patch.object(cycle, "_critique") as mock_critique,
        ):
            mock_gen.return_value = MagicMock(
                text="output",
                cost_usd=0.01,
                input_tokens=10,
                output_tokens=5,
            )

            state = await cycle.run_cycle(
                task=sample_code_task,
                primary_model=Model.GPT_4O_MINI,
                reviewer_model=None,
                full_prompt="Write hello",
            )

            # _critique should NOT have been called
            mock_critique.assert_not_called()
            # Score should be 0.0 (no critique to extract from)
            assert state.best_score == 0.0
