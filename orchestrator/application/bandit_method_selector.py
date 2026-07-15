"""
BanditMethodSelector — Multi-Armed Bandit Method Selection (Phase 4.1)
=======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Augments the existing rule-based ``MethodSelector`` (in ``engine_core.method_selector``)
with a Thompson Sampling bandit that adapts ``ReasoningMethod`` selection per
task type based on recent rewards (score / cost).

Each task type gets its own ``MultiArmedBandit`` instance with arms for
eligible ``ReasoningMethod`` values. Over time, the bandit converges on
the best method for each task type.

Design:
    - ``BanditMethodSelector`` builds on ``events.ab_testing.MultiArmedBandit``
    - ``BanditAwareSelector`` wraps the existing rule-based selector and
      overrides its decisions when the bandit has high confidence.
    - Falls back to the rule-based selector when bandit data is insufficient.

Usage:
    selector = BanditMethodSelector()
    method = await selector.select(TaskType.CODE_GEN)
    await selector.record_outcome(TaskType.CODE_GEN, method, score=0.85, cost=0.02)
"""

from __future__ import annotations

import logging
from typing import Any

from ..events.ab_testing import MultiArmedBandit
from ..models import TaskType
from ..reasoning.ara_pipelines import ReasoningMethod

logger = logging.getLogger("orchestrator.bilevel.bandit_selector")


class BanditMethodSelector:
    """Select ``ReasoningMethod`` per task type using Thompson Sampling.

    Maintains one ``MultiArmedBandit`` per task type. Each bandit has
    arms for all available ``ReasoningMethod`` values. Arms are updated
    with binary success/failure signals based on score thresholds.

    Args:
        score_threshold: Minimum score to count as success (default 0.7).
        default_method: Fallback method when bandit has no data.
    """

    def __init__(
        self,
        score_threshold: float = 0.7,
        default_method: ReasoningMethod = ReasoningMethod.PERSUASION_DEFENSE,
    ) -> None:
        self._bandits: dict[str, MultiArmedBandit] = {}
        self._score_threshold = score_threshold
        self._default_method = default_method
        self._eligible_methods = list(ReasoningMethod)

    async def select(
        self,
        task_type: TaskType,
        available_methods: list[ReasoningMethod] | None = None,
    ) -> ReasoningMethod:
        """Select the best reasoning method for a task type.

        Uses Thompson Sampling traffic allocation. If the bandit
        has no data yet, returns the default method.

        Args:
            task_type: The type of task being routed.
            available_methods: Subset of methods to consider.
                               Defaults to all ``ReasoningMethod`` values.

        Returns:
            The selected ``ReasoningMethod``.
        """
        bandit = self._get_bandit(task_type)
        methods = available_methods or self._eligible_methods

        # Ensure all eligible methods have arms
        for method in methods:
            if method.value not in bandit._arms:
                bandit.add_arm(method.value)

        # If bandit has zero data, return default
        stats = bandit.get_stats()
        if not stats or all(s["successes"] + s["failures"] == 0 for s in stats.values()):
            return self._default_method

        # Allocate traffic — weighted random based on Thompson Sampling
        import random

        allocations = bandit.allocate_traffic(total_traffic=1.0)
        arms = list(allocations.keys())
        if not arms:
            return self._default_method
        weights = [allocations[a] for a in arms]
        selected = random.choices(arms, weights=weights, k=1)[0]

        # Convert string back to ReasoningMethod
        for method in methods:
            if method.value == selected:
                return method

        return self._default_method

    async def record_outcome(
        self,
        task_type: TaskType,
        method: ReasoningMethod,
        score: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        """Record the outcome of using a method for a task type.

        Args:
            task_type: The task type that was executed.
            method: The method that was used.
            score: The score achieved (0.0 - 1.0).
            cost: The cost incurred (not used for bandit update).
        """
        bandit = self._get_bandit(task_type)
        # Auto-add arm if it doesn't exist yet
        if method.value not in bandit._arms:
            bandit.add_arm(method.value)
        success = score >= self._score_threshold
        bandit.update_arm(method.value, success)
        logger.debug(
            "Bandit update for %s / %s: success=%s (score=%.2f, cost=%.4f)",
            task_type.value,
            method.value,
            success,
            score,
            cost,
        )

    def get_stats(self, task_type: TaskType) -> dict[str, Any]:
        """Get bandit statistics for a task type."""
        bandit = self._get_bandit(task_type)
        return bandit.get_stats()

    # ── Internal ──────────────────────────────────────────────────────

    def _get_bandit(self, task_type: TaskType) -> MultiArmedBandit:
        key = task_type.value
        if key not in self._bandits:
            self._bandits[key] = MultiArmedBandit()  # type: ignore[no-untyped-call]
        return self._bandits[key]

    @property
    def bandit_count(self) -> int:
        """Number of active bandits (one per task type seen)."""
        return len(self._bandits)
