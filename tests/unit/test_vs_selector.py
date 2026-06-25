"""Tests for CandidateSelector — VS candidate selection with quality scoring."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.application.vs_selector import CandidateSelector
from orchestrator.domain.ports import QualityScorer
from orchestrator.application.verbalized_sampling import VSCandidate
from orchestrator.models import Task

# ── Stub scorer implementations ─────────────────────────────────────────────


class StubScorer:
    """Deterministic scorer for testing: returns score from a per-text dict."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.called_with: list[str] = []

    async def score(self, task: object, text: str) -> float:
        self.called_with.append(text)
        return self.scores.get(text, 0.5)


class FailingScorer:
    """Scorer that raises on specific texts."""

    def __init__(self, fail_on: set[str]):
        self.fail_on = fail_on
        self.called_with: list[str] = []

    async def score(self, task: object, text: str) -> float:
        self.called_with.append(text)
        if text in self.fail_on:
            raise RuntimeError(f"Scorer failed on: {text}")
        return 0.8


def make_candidates(texts: list[tuple[str, float]]) -> list[VSCandidate]:
    """Build candidates from (text, probability) pairs."""
    return [VSCandidate(text=t, probability=p) for t, p in texts]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCandidateSelector:
    """CandidateSelector.select() behavior."""

    @pytest.mark.asyncio
    async def test_select_empty_returns_none(self):
        """Empty candidate list returns None."""
        selector = CandidateSelector(scorer=StubScorer({}), prefilter_keep=4)
        result = await selector.select(MagicMock(), [])
        assert result is None

    @pytest.mark.asyncio
    async def test_prefilter_keeps_only_top_n(self):
        """With k=6 candidates and prefilter_keep=4, scorer is called exactly 4 times."""
        candidates = make_candidates(
            [
                ("a", 0.9),
                ("b", 0.8),
                ("c", 0.7),
                ("d", 0.6),
                ("e", 0.5),
                ("f", 0.4),
            ]
        )
        scorer = StubScorer({"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.8})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=4)

        result = await selector.select(MagicMock(), candidates)

        assert result is not None
        assert len(scorer.called_with) == 4  # only top-4 scored
        assert "e" not in scorer.called_with  # bottom-2 filtered
        assert "f" not in scorer.called_with

    @pytest.mark.asyncio
    async def test_argmax_score_returned(self):
        """Highest-scored candidate is selected."""
        candidates = make_candidates(
            [
                ("a", 0.9),
                ("b", 0.8),
                ("c", 0.7),
            ]
        )
        # c scores highest despite lowest probability
        scorer = StubScorer({"a": 0.5, "b": 0.6, "c": 0.9})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=3)

        result = await selector.select(MagicMock(), candidates)

        assert result is not None
        assert result.text == "c"

    @pytest.mark.asyncio
    async def test_tie_goes_to_higher_probability(self):
        """Tied scores are broken by probability."""
        candidates = make_candidates(
            [
                ("a", 0.9),  # higher probability
                ("b", 0.6),  # lower probability, same score
            ]
        )
        scorer = StubScorer({"a": 0.7, "b": 0.7})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=2)

        result = await selector.select(MagicMock(), candidates)

        assert result is not None
        assert result.text == "a"  # higher probability wins tie

    @pytest.mark.asyncio
    async def test_scorer_failure_does_not_crash(self):
        """A scorer that raises on one candidate doesn't crash the whole selection."""
        candidates = make_candidates(
            [
                ("ok", 0.9),
                ("fail", 0.8),
                ("also_ok", 0.7),
            ]
        )
        scorer = FailingScorer(fail_on={"fail"})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=3)

        result = await selector.select(MagicMock(), candidates)

        # Should return whichever surviving candidate scored highest
        assert result is not None
        assert result.text in ("ok", "also_ok")

    @pytest.mark.asyncio
    async def test_all_scorers_fail_returns_none(self):
        """When every scorer call raises, select returns None."""
        candidates = make_candidates([("a", 0.9), ("b", 0.8)])
        scorer = FailingScorer(fail_on={"a", "b"})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=2)

        result = await selector.select(MagicMock(), candidates)

        assert result is None

    @pytest.mark.asyncio
    async def test_single_candidate_scores_directly(self):
        """A single candidate is scored and returned."""
        candidates = make_candidates([("only", 0.5)])
        scorer = StubScorer({"only": 0.95})
        selector = CandidateSelector(scorer=scorer, prefilter_keep=4)

        result = await selector.select(MagicMock(), candidates)

        assert result is not None
        assert result.text == "only"
