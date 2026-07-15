"""
Verifier Protocol — abstract scoring port for objective verification.

All concrete verifiers implement this Protocol so that
``ModelCascader`` / ``EnhancedSelfConsistencyStage`` can depend on the
abstraction without importing concretes (import-linter Rule).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from orchestrator.models import TaskType, Verdict


@runtime_checkable
class Verifier(Protocol):
    """Protocol for objective verifiers that score generated responses.

    Implementations must be async and accept ``prompt``, ``response``,
    and ``task_type``.  They return a ``Verdict`` with:
      ``passed``  — boolean decision
      ``score``   — 0..1 float (for cascade min_score comparison)
      ``signals`` — which sub-checks fired (telemetry)
      ``detail``  — human-readable explanation
    """

    async def verify(
        self,
        *,
        prompt: str,
        response: str,
        task_type: TaskType,
    ) -> Verdict:
        """Score *response* against objective criteria.

        Args:
            prompt: The original prompt.
            response: The generated response to verify.
            task_type: Type of task that produced the response.

        Returns:
            Verdict with pass/fail, score, signals, and detail.
        """
        ...
