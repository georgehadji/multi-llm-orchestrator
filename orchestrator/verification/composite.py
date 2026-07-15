"""
Composite Verifier — combines multiple verifiers into a single verdict.

Uses configurable weighting strategy to blend sub-verifier scores.
Fail-closed: if any sub-verifier raises, the composite returns
``Verdict(passed=False)``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from orchestrator.models import TaskType, Verdict

from .port import Verifier

logger = logging.getLogger(__name__)


class CompositeVerifier:
    """Weighted combination of multiple Verifier instances.

    Each sub-verifier contributes its ``score * weight`` to the
    composite score.  If any sub-verifier returns ``passed=False``
    and its weight is above ``hard_gate_weight``, the composite
    also returns ``passed=False``.

    Default strategy: equal-weight average.

    Usage:
        composite = CompositeVerifier([
            (ast_verifier, 0.5),
            (regex_verifier, 0.3),
            (json_verifier, 0.2),
        ], hard_gate_weight=0.3)
        verdict = await composite.verify(prompt="...", response="...", task_type=...)
    """

    def __init__(
        self,
        verifiers: Sequence[tuple[Verifier, float]] = (),
        *,
        hard_gate_weight: float = 0.3,
    ) -> None:
        """Initialize composite verifier.

        Args:
            verifiers: Sequence of (verifier, weight) tuples.
                Weights are normalised to 1.0 internally.
            hard_gate_weight: Minimum weight threshold for a failing
                sub-verifier to force ``passed=False``.  Prevents a
                low-weight trivial check from vetoing the whole result.
        """
        if not verifiers:
            self._verifiers: list[tuple[Verifier, float]] = []
            self._weights_normalized: list[float] = []
        else:
            self._verifiers = list(verifiers)
            raw_weights = [w for _, w in verifiers]
            total = sum(raw_weights) or 1.0
            self._weights_normalized = [w / total for w in raw_weights]

        self._hard_gate_weight = hard_gate_weight

    @property
    def verifier_count(self) -> int:
        """Number of registered sub-verifiers."""
        return len(self._verifiers)

    async def verify(
        self,
        *,
        prompt: str,
        response: str,
        task_type: TaskType,
    ) -> Verdict:
        """Run all sub-verifiers and aggregate their verdicts."""
        if not self._verifiers:
            return Verdict(
                passed=True,
                score=0.5,
                signals=("composite_empty",),
                detail="No verifiers registered",
            )

        all_signals: list[str] = []
        total_score = 0.0
        failures: list[str] = []
        has_hard_failure = False

        for (verifier, _), weight in zip(self._verifiers, self._weights_normalized):
            try:
                verdict = await verifier.verify(
                    prompt=prompt,
                    response=response,
                    task_type=task_type,
                )
            except Exception as exc:
                logger.warning("Sub-verifier %s raised: %s", type(verifier).__name__, exc)
                verdict = Verdict(
                    passed=False,
                    score=0.0,
                    signals=(),
                    detail=str(exc),
                )

            all_signals.extend(verdict.signals)
            total_score += verdict.score * weight

            if not verdict.passed and weight >= self._hard_gate_weight:
                has_hard_failure = True
                failures.append(f"{type(verifier).__name__}: {verdict.detail or 'failed'}")

        passed = not has_hard_failure
        score = min(1.0, max(0.0, total_score))

        return Verdict(
            passed=passed,
            score=score,
            signals=tuple(all_signals),
            detail="; ".join(failures) if failures else "All verifiers passed",
        )
