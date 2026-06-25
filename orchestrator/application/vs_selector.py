"""
CandidateSelector — VS candidate selection with quality scoring.
===================================================================
Part of Application Layer.

Selects the best candidate from a list of VSCandidate objects by:
1. Free probability prefilter (bounds LLM scoring cost)
2. Quality scoring via injected QualityScorer port
3. Argmax with tiebreak on probability

Usage:
    selector = CandidateSelector(scorer=EvaluatorScorer(evaluator))
    best = await selector.select(task, candidates)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.ports import QualityScorer
    from ..application.verbalized_sampling import VSCandidate


class CandidateSelector:
    """Selects the best candidate using an injectable quality scorer.

    Args:
        scorer: QualityScorer port — callers provide an adapter
            (e.g., EvaluatorScorer, ProbabilityScorer).
        prefilter_keep: Number of top-probability candidates to keep
            before LLM scoring. Limits cost.
    """

    def __init__(
        self,
        scorer: QualityScorer,
        prefilter_keep: int = 4,
    ):
        self._scorer = scorer
        self._prefilter_keep = prefilter_keep

    async def select(
        self,
        task: object,
        candidates: list[VSCandidate],
    ) -> VSCandidate | None:
        """Select the best candidate from a list.

        1. Empty → None.
        2. Sort by probability desc, keep top ``prefilter_keep``.
        3. Score survivors via ``scorer.score(task, text)`` (parallel).
        4. Return argmax score; tie-break by probability.
        """
        if not candidates:
            return None

        # 1. Free prefilter: sort by probability, keep top N
        sorted_cands = sorted(candidates, key=lambda c: c.probability, reverse=True)
        survivors = sorted_cands[: self._prefilter_keep]

        # 2. Score survivors in parallel
        async def _score(cand: VSCandidate) -> tuple[float | None, VSCandidate]:
            """Score a candidate, returning (score, candidate).

            Returns None for score if the scorer raised.
            """
            try:
                score = await self._scorer.score(task, cand.text)
                return (score, cand)
            except Exception:
                return (None, cand)

        scored = await asyncio.gather(*[_score(c) for c in survivors])

        # Filter out failed scores
        valid = [(s, c) for s, c in scored if s is not None]
        if not valid:
            return None

        # 3. Argmax score; tie-break by probability
        best = max(valid, key=lambda pair: (pair[0], pair[1].probability))

        return best[1]
