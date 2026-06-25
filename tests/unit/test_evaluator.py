"""Unit tests for EvaluatorService — score parsing, aggregation, evaluation flow."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from orchestrator.application.evaluator import EvaluatorService
from orchestrator.models import Task, TaskType, Model

# ═══════════════════════════════════════════════════════════════════════════════
# parse_score — pure function
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseScore:
    """EvaluatorService.parse_score() normalizes LLM output to [0.0, 1.0]."""

    def test_json_score(self):
        assert EvaluatorService.parse_score('{"score": 0.85}') == 0.85

    def test_json_capital_key(self):
        assert EvaluatorService.parse_score('{"Score": 0.72}') == 0.72

    def test_fenced_json(self):
        assert EvaluatorService.parse_score('```json\n{"score": 0.77}\n```') == 0.77

    def test_inline_score_label(self):
        assert EvaluatorService.parse_score("The score: 0.75") == 0.75

    def test_x_out_of_10(self):
        assert EvaluatorService.parse_score("8/10") == 0.8

    def test_x_out_of_100(self):
        assert EvaluatorService.parse_score("85/100") == 0.85

    def test_percentage(self):
        assert EvaluatorService.parse_score("Score: 92%") == 0.92

    def test_bare_float(self):
        assert EvaluatorService.parse_score("0.82") == 0.82

    def test_rating_label(self):
        assert EvaluatorService.parse_score("rating: 0.63") == 0.63

    def test_negative_clamped(self):
        assert EvaluatorService.parse_score('{"score": -0.3}') == 0.0

    def test_over_one_clamped(self):
        assert EvaluatorService.parse_score('{"score": 3.5}') == 1.0

    def test_gibberish_default(self):
        assert EvaluatorService.parse_score("Unparseable text.") == 0.5

    def test_empty_default(self):
        assert EvaluatorService.parse_score("") == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# _aggregate
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregate:
    """EvaluatorService._aggregate() self-consistency logic."""

    def test_average_when_consistent(self):
        e = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])
        assert e._aggregate([0.80, 0.82], "t1") == 0.81

    def test_lower_when_inconsistent(self):
        e = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])
        assert e._aggregate([0.95, 0.70], "t1") == 0.70

    def test_single_score(self):
        e = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])
        assert e._aggregate([0.88], "t1") == 0.88

    def test_empty_returns_default(self):
        e = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])
        assert e._aggregate([], "t1") == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate — full flow
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluate:
    """EvaluatorService.evaluate() full scoring flow."""

    @pytest.fixture
    def mock_client(self):
        c = MagicMock()
        c.call = AsyncMock(
            return_value=MagicMock(
                text='{"score": 0.85, "issues": []}',
                cost_usd=0.001,
                input_tokens=50,
                output_tokens=30,
            )
        )
        return c

    @pytest.fixture
    def mock_budget(self):
        b = MagicMock()
        b.charge = AsyncMock()
        return b

    @pytest.fixture
    def get_models_fn(self):
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

    @pytest.fixture
    def sample_task(self):
        return Task(
            id="eval-1", type=TaskType.CODE_GEN, prompt="Write a function", max_output_tokens=4096
        )

    @pytest.mark.asyncio
    async def test_returns_critique_report(self, evaluator, sample_task):
        r = await evaluator.evaluate(sample_task, "def foo(): pass")
        assert r.score == 0.85
        assert r.task_id == "eval-1"
        assert r.items == []

    @pytest.mark.asyncio
    async def test_charges_budget(self, evaluator, sample_task, mock_budget):
        await evaluator.evaluate(sample_task, "output")
        assert mock_budget.charge.call_count == 2

    @pytest.mark.asyncio
    async def test_no_models_fallback(self, sample_task):
        e = EvaluatorService(MagicMock(), MagicMock(), lambda x: [])
        r = await e.evaluate(sample_task, "output")
        assert r.score == 0.5

    @pytest.mark.asyncio
    async def test_client_error_fallback(self, evaluator, sample_task, mock_client):
        mock_client.call.side_effect = RuntimeError("API timeout")
        r = await evaluator.evaluate(sample_task, "output")
        assert r.score == 0.5

    @pytest.mark.asyncio
    async def test_parses_issues(self, mock_client, mock_budget, get_models_fn, sample_task):
        mock_client.call.return_value.text = json.dumps(
            {
                "score": 0.75,
                "issues": [
                    {"severity": "major", "category": "security", "description": "SQL injection"}
                ],
            }
        )
        e = EvaluatorService(client=mock_client, budget=mock_budget, get_models_fn=get_models_fn)
        r = await e.evaluate(sample_task, "def foo(): pass")
        assert len(r.items) == 1
        assert r.items[0].description == "SQL injection"
        assert r.items[0].severity.value == "major"

    @pytest.mark.asyncio
    async def test_parses_issues_from_fenced_response(
        self, mock_client, mock_budget, get_models_fn, sample_task
    ):
        mock_client.call.return_value.text = '```json\n{"score": 0.88, "issues": []}\n```'
        e = EvaluatorService(client=mock_client, budget=mock_budget, get_models_fn=get_models_fn)
        r = await e.evaluate(sample_task, "output")
        assert r.score == 0.88
