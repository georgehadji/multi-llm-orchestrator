"""Unit tests for orchestrator.services.evaluator.EvaluatorService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.api_clients import APIResponse
from orchestrator.models import Model, Task, TaskStatus, TaskType
from orchestrator.services.evaluator import EvaluatorService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _task() -> Task:
    return Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="Write hello world",
        context="",
        dependencies=[],
        acceptance_threshold=0.8,
    )


def _api_response(text: str, cost: float = 0.001) -> APIResponse:
    r = APIResponse(
        text=text,
        input_tokens=10,
        output_tokens=5,
        model=Model.GPT_4O_MINI,
    )
    r.cost_usd = cost
    return r


def _make_service(responses: list[str] | None = None) -> EvaluatorService:
    """Build an EvaluatorService with mocked client, budget, and model list."""
    client = MagicMock()
    if responses:
        client.call = AsyncMock(
            side_effect=[_api_response(r) for r in responses]
        )
    else:
        client.call = AsyncMock(return_value=_api_response('{"score": 0.85}'))

    budget = MagicMock()
    budget.charge = AsyncMock()

    def get_models(task_type: TaskType) -> list[Model]:
        return [Model.GPT_4O_MINI]

    return EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=get_models,
    )


# ─────────────────────────────────────────────────────────────────────────────
# evaluate()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluate_returns_score():
    svc = _make_service(['{"score": 0.9}', '{"score": 0.88}'])
    score = await svc.evaluate(_task(), "good output")
    assert 0.88 <= score <= 0.90


@pytest.mark.asyncio
async def test_evaluate_charges_budget_per_run():
    svc = _make_service(['{"score": 0.8}', '{"score": 0.82}'])
    await svc.evaluate(_task(), "output")
    assert svc._budget.charge.call_count == 2


@pytest.mark.asyncio
async def test_evaluate_returns_05_when_no_models():
    client = MagicMock()
    budget = MagicMock()
    svc = EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=lambda _: [],  # no models
    )
    score = await svc.evaluate(_task(), "output")
    assert score == 0.5


@pytest.mark.asyncio
async def test_evaluate_falls_back_on_api_error():
    client = MagicMock()
    client.call = AsyncMock(side_effect=RuntimeError("API down"))
    budget = MagicMock()
    budget.charge = AsyncMock()
    svc = EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=lambda _: [Model.GPT_4O_MINI],
    )
    score = await svc.evaluate(_task(), "output")
    assert score == 0.5  # fallback from all-failed runs


@pytest.mark.asyncio
async def test_evaluate_uses_lower_score_on_inconsistency():
    # scores differ by more than 0.05 → use min
    svc = _make_service(['{"score": 0.9}', '{"score": 0.7}'])
    score = await svc.evaluate(_task(), "output")
    assert score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_evaluate_averages_consistent_scores():
    svc = _make_service(['{"score": 0.8}', '{"score": 0.82}'])
    score = await svc.evaluate(_task(), "output")
    assert score == pytest.approx(0.81, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# parse_score() — static, pure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"score": 0.85}', 0.85),
        ('{"score": 0.85, "reasoning": "good"}', 0.85),
        ('{"Score": 0.9}', 0.9),
        ("0.75", 0.75),
        ("score: 0.6", 0.6),
        ("score=0.7", 0.7),
        ("85%", 0.85),
        ("8.5/10", 0.85),
        ("70/100", 0.70),
        ("7 out of 10", 0.7),
        ("out of 10: 6", 0.6),
        ("rating: 0.8", 0.8),
        ("评分: 0.9", 0.9),
        # Markdown fences
        ("```json\n{\"score\": 0.95}\n```", 0.95),
        # Clamping
        ('{"score": 1.5}', 1.0),
        ('{"score": -0.1}', 0.0),
        # Fallback
        ("completely unparseable garbage !!!!", 0.5),
    ],
)
def test_parse_score_parametrized(text: str, expected: float):
    result = EvaluatorService.parse_score(text)
    assert result == pytest.approx(expected, abs=0.01)


def test_parse_score_bare_float():
    assert EvaluatorService.parse_score("The quality is 0.92 out of 1.") == pytest.approx(0.92)


def test_parse_score_clamps_to_zero_one():
    assert EvaluatorService.parse_score('{"score": 99}') == 1.0
    assert EvaluatorService.parse_score('{"score": -5}') == 0.0
