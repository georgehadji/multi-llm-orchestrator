"""EvaluateStage must not re-divide an already-0..1 score by 10.

Root cause of the near-zero task scores in dogfooding: the EvaluatorService
returns a CritiqueReport.score already in [0.0, 1.0] (per its docstring and the
0.5 default), but EvaluateStage did `ctx.score = critique_report.score / 10.0`,
turning a genuine 0.85 into 0.085 and the 0.5 default into 0.05 — exactly the
0.05/0.005/0.09 values observed.
"""

import asyncio
import types

import pytest

pytestmark = pytest.mark.unit

from orchestrator.engine_core.stages.evaluate import EvaluateStage

pytestmark = pytest.mark.unit


class _FakeReport:
    def __init__(self, score: float) -> None:
        self.score = score

    def to_prompt_context(self, max_items: int = 10) -> str:
        return "critique"


class _FakeEvaluator:
    def __init__(self, score: float) -> None:
        self._score = score

    async def evaluate(self, task, output):  # noqa: ANN001
        return _FakeReport(self._score)


def _run(score_in: float) -> float:
    ctx = types.SimpleNamespace(
        output="def add(a, b): return a + b",
        score=0.0,
        critique="",
        task=types.SimpleNamespace(id="t1"),
    )
    stage = EvaluateStage(_FakeEvaluator(score_in))
    asyncio.run(stage.process(ctx))
    return ctx.score


def test_high_score_passes_through_unscaled():
    assert _run(0.85) == pytest.approx(0.85)  # was 0.085


def test_default_midscore_not_collapsed_to_near_zero():
    assert _run(0.5) == pytest.approx(0.5)  # was 0.05


def test_perfect_score_preserved():
    assert _run(1.0) == pytest.approx(1.0)  # was 0.1
