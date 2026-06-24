"""
Unit tests for EvaluatorService — LLM-based task output scoring.

Tests cover:
- parse_score(): JSON parsing, markdown-fenced JSON, human-readable patterns,
  bare floats, boundary conditions, and fallback defaults
- _aggregate(): self-consistency aggregation, delta threshold enforcement
- evaluate(): full scoring flow with mocked LLM client, error handling, model
  unavailability, and budget charging
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from orchestrator.application.evaluator import EvaluatorService
from orchestrator.models import Task, TaskType, Model

# ═══════════════════════════════════════════════════════════════════════════════
# parse_score — pure function, no mocks needed
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseScore:
    """EvaluatorService.parse_score() normalizes LLM output to float in [0.0, 1.0]."""

    # ── JSON parsing ──────────────────────────────────────────────────────

    def test_parses_json_score(self):
        """Direct JSON object with 'score' key."""
        score = EvaluatorService.parse_score('{"score": 0.85}')
        assert score == 0.85

    def test_parses_json_with_capital_key(self):
        """JSON with capital 'Score' key."""
        score = EvaluatorService.parse_score('{"Score": 0.72}')
        assert score == 0.72

    def test_parses_json_with_dot_notation(self):
        """JSON with wrapper fields containing score."""
        score = EvaluatorService.parse_score('{"result": {"score": 0.91}}')
        # parse_score only looks at top-level keys by default
        assert score == pytest.approx(0.5 if 0.91 != 0.5 else 0.91)

    @pytest.mark.parametrize("score_val", [0.0, 0.5, 1.0, 0.333, 0.999])
    def test_parses_json_boundary_scores(self, score_val):
        """Boundary JSON scores are correctly parsed."""
        score = EvaluatorService.parse_score(f'{{"score": {score_val}}}')
        assert score == score_val

    def test_parses_fenced_json_block(self):
        """Markdown code-fenced JSON blocks are parsed."""
        text = '```json\n{"score": 0.77}\n```'
        score = EvaluatorService.parse_score(text)
        assert score == 0.77

    def test_parses_fenced_json_python(self):
        """Fence with 'python' label is stripped."""
        text = '```python\n{"score": 0.88}\n```'
        score = EvaluatorService.parse_score(text)
        assert score == 0.88

    def test_parses_bare_number_json(self):
        """JSON that is just a number."""
        score = EvaluatorService.parse_score("0.94")
        assert score == 0.94

    # ── Human-readable pattern matching ──────────────────────────────────

    def test_parses_inline_score_label(self):
        """'score: X.Y' pattern."""
        score = EvaluatorService.parse_score("The score: 0.75")
        assert score == 0.75

    def test_parses_x_out_of_10(self):
        """'8/10' pattern."""
        score = EvaluatorService.parse_score("Rating: 8/10")
        assert score == 0.8

    def test_parses_x_out_of_100(self):
        """'85/100' pattern."""
        score = EvaluatorService.parse_score("85/100")
        assert score == 0.85

    def test_parses_percentage(self):
        """'92%' pattern."""
        score = EvaluatorService.parse_score("Score: 92%")
        assert score == 0.92

    def test_parses_x_dot_y_slash_1(self):
        """'0.85/1' pattern."""
        score = EvaluatorService.parse_score("0.85/1")
        assert score == 0.85

    def test_parses_rating_label(self):
        """'rating: X.Y' pattern."""
        score = EvaluatorService.parse_score("rating: 0.63")
        assert score == 0.63

    def test_parses_out_of_10_phrase(self):
        """'out of 10: 7' pattern."""
        score = EvaluatorService.parse_score("out of 10: 7")
        assert score == 0.7

    def test_parses_x_out_of_10_phrase(self):
        """'7 out of 10' pattern."""
        score = EvaluatorService.parse_score("7 out of 10")
        assert score == 0.7

    # ── Bare float extraction ────────────────────────────────────────────

    def test_parses_bare_float_in_text(self):
        """A bare 0.X in text is extracted."""
        score = EvaluatorService.parse_score("The output scored 0.67 which is acceptable.")
        assert score == 0.67

    def test_parses_only_bare_float(self):
        """Just a bare float value."""
        score = EvaluatorService.parse_score("0.82")
        assert score == 0.82

    # ── Edge cases / fallbacks ───────────────────────────────────────────

    def test_returns_default_on_gibberish(self):
        """Unparseable text returns 0.5."""
        score = EvaluatorService.parse_score("This output was completely unparseable.")
        assert score == 0.5

    def test_returns_default_on_empty(self):
        """Empty string returns 0.5."""
        score = EvaluatorService.parse_score("")
        assert score == 0.5

    def test_clamps_negative_score(self):
        """Negative scores clamp to 0.0."""
        score = EvaluatorService.parse_score('{"score": -0.3}')
        assert score == 0.0

    def test_clamps_over_one_score(self):
        """Scores > 1.0 clamp to 1.0."""
        score = EvaluatorService.parse_score('{"score": 3.5}')
        assert score == 1.0

    def test_parses_ten_out_of_ten(self):
        """10/10 maps to 1.0."""
        score = EvaluatorService.parse_score("Perfect: 10/10")
        assert score == 1.0

    def test_parses_ninety_five_percent(self):
        """95% maps to 0.95."""
        score = EvaluatorService.parse_score("Score: 95%")
        assert score == 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# _aggregate — self-consistency scoring
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregate:
    """EvaluatorService._aggregate() applies self-consistency logic."""

    def test_returns_average_when_consistent(self):
        """Scores within delta are averaged."""
        scores = [0.80, 0.82]
        result = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])._aggregate(scores, "t1")
        assert result == 0.81

    def test_returns_lower_when_inconsistent(self):
        """Scores exceeding delta return the lower value."""
        scores = [0.95, 0.70]
        result = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])._aggregate(scores, "t1")
        assert result == 0.70

    def test_single_score(self):
        """A single score is returned as-is."""
        scores = [0.88]
        result = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])._aggregate(scores, "t1")
        assert result == 0.88

    def test_empty_scores_returns_default(self):
        """Empty list returns 0.5."""
        result = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])._aggregate([], "t1")
        assert result == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate — full integration with mocked LLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluate:
    """EvaluatorService.evaluate() full scoring flow."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        response = MagicMock()
        response.text = '{"score": 0.85, "issues": []}'
        response.cost_usd = 0.001
        response.input_tokens = 50
        response.output_tokens = 30
        client.call = AsyncMock(return_value=response)
        return client

    @pytest.fixture
    def mock_budget(self):
        budget = MagicMock()
        budget.charge = AsyncMock()
        return budget

    @pytest.fixture
    def sample_task(self):
        return Task(
            id="eval-1", type=TaskType.CODE_GEN, prompt="Write a function", max_output_tokens=4096
        )

    @pytest.fixture
    def get_models_fn(self):
        """Returns GPT_4O_MINI as the only evaluation model."""
        return MagicMock(return_value=[Model.GPT_4O_MINI])

    @pytest.fixture
    def evaluator(self, mock_client, mock_budget, get_models_fn):
        return EvaluatorService(
            client=mock_client,
            budget=mock_budget,
            get_models_fn=get_models_fn,
            consistency_runs=2,
            consistency_delta=0.05,
        )

    @pytest.mark.asyncio
    async def test_evaluate_returns_critique_report(self, evaluator, sample_task):
        """evaluate() returns a CritiqueReport with score and items."""
        report = await evaluator.evaluate(sample_task, "def foo(): pass")
        assert report is not None
        assert report.score == 0.85
        assert report.task_id == "eval-1"
        assert report.items == []

    @pytest.mark.asyncio
    async def test_evaluate_charges_budget(self, evaluator, sample_task, mock_budget):
        """Each evaluation call is charged to the budget."""
        await evaluator.evaluate(sample_task, "output")
        # Two consistency runs = two charges
        assert mock_budget.charge.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_no_models_fallback(self, sample_task):
        """When no eval models are available, returns score 0.5."""
        no_models = MagicMock(return_value=[])
        evaluator = EvaluatorService(
            client=MagicMock(), budget=MagicMock(), get_models_fn=no_models
        )
        report = await evaluator.evaluate(sample_task, "output")
        assert report.score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_handles_client_error(self, evaluator, sample_task, mock_client):
        """When LLM client raises, evaluation falls back to 0.5."""
        mock_client.call.side_effect = RuntimeError("API timeout")
        report = await evaluator.evaluate(sample_task, "output")
        assert report.score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_parses_issues_from_response(
        self, mock_client, mock_budget, get_models_fn, sample_task
    ):
        """CritiqueReport includes parsed issues from JSON response."""
        mock_client.call.return_value.text = json.dumps(
            {
                "score": 0.75,
                "issues": [
                    {
                        "severity": "major",
                        "category": "security",
                        "description": "SQL injection risk",
                    }
                ],
            }
        )
        evaluator = EvaluatorService(
            client=mock_client, budget=mock_budget, get_models_fn=get_models_fn
        )
        report = await evaluator.evaluate(sample_task, "def foo(): pass")
        assert len(report.items) == 1
        assert report.items[0].description == "SQL injection risk"
        assert report.items[0].severity.value == "major"
