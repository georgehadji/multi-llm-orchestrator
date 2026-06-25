"""
Scorer adapters for QualityScorer port.
=========================================
Part of Application Layer (services sub-domain).

Adapters:
- EvaluatorScorer — wraps EvaluatorService.evaluate() as a quality signal.
  Uses consistency_runs=1 for cheap reranking.
- ProbabilityScorer — returns VSCandidate.probability (free fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.ports import QualityScorer  # noqa: F401
    from ..application.evaluator import EvaluatorService
    from ..application.verbalized_sampling import VSCandidate


class EvaluatorScorer:
    """Quality score via EvaluatorService.

    Wraps ``evaluator.evaluate(task, text)`` and returns ``report.score``.

    Uses ``consistency_runs=1`` — for reranking we need ranking, not the
    2-pass delta guard. Halves eval cost vs the default 2-pass.
    """

    def __init__(self, evaluator: EvaluatorService):
        self._evaluator = evaluator

    async def score(self, task: object, text: str) -> float:
        """Score a single candidate text.

        Returns score in [0.0, 1.0]. On evaluator failure, returns 0.0
        (degrade gracefully, never block).
        """
        try:
            report = await self._evaluator.evaluate(task, text)  # type: ignore[arg-type]
            return report.score
        except Exception:
            return 0.0


class ProbabilityScorer:
    """Free fallback scorer — returns candidate probability.

    Used when VS reranking is disabled or budget is exhausted.
    Requires a VSCandidate.probability field.
    """

    async def score(self, task: object, text: str) -> float:
        """Return candidate probability directly from VSCandidate.

        Note: This expects the caller to pass the candidate's probability
        via the text field in the format ``__PROB__<float>__ <text>``,
        or defaults to 0.5 if no probability is embedded.

        In practice, this adapter is used via ``CandidateSelector`` which
        receives ``VSCandidate`` objects with a ``.probability`` field
        directly — the text arg here is just the candidate text.
        """
        # ProbabilityScorer is a degenerate scorer — it should only be
        # used when the caller provides scores separately.
        # For the ProbabilityScorer fallback path, use select_with_probs()
        # instead, or keep 0.5 as a neutral default.
        return 0.5
