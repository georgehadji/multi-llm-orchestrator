"""
OrthogonalExploration — Diversity Forcing for Stagnant Search
===============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

When the search pipeline detects stagnation (flat score curves, high repetition),
``OrthogonalExploration`` forces diverse interventions by ensuring consecutive
proposals differ along at least two of the following axes:

- **Model** (provider/ID)
- **Temperature** (low vs high)
- **Skill prefix** (specialised vs generic)
- **ARA method** (reasoning strategy)

This prevents the system from getting stuck in local optima by repeatedly
trying similar (model, temperature, method) combinations.

Usage:
    explorer = OrthogonalExploration()
    next_config = explorer.propose_diverse(
        current={"model": "gpt4", "temperature": 0.3, "method": "sot"},
        stagnation_score=0.8,
    )
    # next_config will differ in at least 2 axes from current
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.bilevel.orthogonal_exploration")


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class ExplorationConfig:
    """Available values for each exploration axis."""

    models: list[str] = field(
        default_factory=lambda: [
            "gpt4",
            "gpt4-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "gemini-pro",
            "deepseek-chat",
        ]
    )
    temperatures: list[float] = field(
        default_factory=lambda: [
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ]
    )
    skill_prefixes: list[str] = field(
        default_factory=lambda: [
            "general",
            "code",
            "math",
            "creative",
            "analysis",
        ]
    )
    methods: list[str] = field(
        default_factory=lambda: [
            "sot",
            "tot",
            "persuasion_defense",
            "debate",
            "scientific",
        ]
    )


@dataclass
class Intervention:
    """A diverse intervention proposal."""

    model: str = ""
    temperature: float = 0.5
    skill_prefix: str = "general"
    method: str = "persuasion_defense"
    diversity_score: int = 0  # number of axes changed from previous

    @property
    def axis_count(self) -> int:
        """Number of configured axes (always 4)."""
        return 4


# ── Explorer ──────────────────────────────────────────────────────────────


class OrthogonalExploration:
    """Force diverse interventions when stagnation is detected.

    Maintains a history of recent proposals to ensure consecutive
    interventions differ on at least 2 of 4 axes (model, temperature,
    skill_prefix, method).

    Args:
        config: Available values for each axis. Defaults provide good spread.
        min_axes_changed: Minimum number of axes that must differ (default 2).
        history_size: How many past interventions to track (default 5).
    """

    def __init__(
        self,
        config: ExplorationConfig | None = None,
        min_axes_changed: int = 2,
        history_size: int = 5,
    ) -> None:
        self._config = config or ExplorationConfig()
        self._min_axes_changed = min_axes_changed
        self._history: list[Intervention] = []
        self._history_size = history_size

    def propose_diverse(
        self,
        current: dict[str, Any] | None = None,
        stagnation_score: float = 0.0,
        forced: bool = False,
    ) -> Intervention:
        """Propose a diverse intervention.

        When stagnation is high or ``forced=True``, the proposal is
        guaranteed to differ on at least ``min_axes_changed`` axes
        from the most recent intervention.

        Args:
            current: Current configuration dict with optional keys
                     ``model``, ``temperature``, ``skill_prefix``, ``method``.
            stagnation_score: Current stagnation score (0.0 - 1.0).
            forced: Force diversity regardless of stagnation score.

        Returns:
            An ``Intervention`` with diverse axis values.
        """
        # Build a candidate that differs from the last intervention
        last = self._history[-1] if self._history else None

        candidate = self._random_intervention()
        if current:
            candidate = Intervention(
                model=current.get("model", candidate.model),
                temperature=current.get("temperature", candidate.temperature),
                skill_prefix=current.get("skill_prefix", candidate.skill_prefix),
                method=current.get("method", candidate.method),
            )

        # If stagnation or forced, ensure diversity from last
        if (stagnation_score > 0.3 or forced) and last is not None:
            candidate = self._ensure_diversity(candidate, last)

        # Track history
        self._history.append(candidate)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # Compute diversity score from previous
        if last is not None:
            candidate.diversity_score = self._count_axes_changed(candidate, last)

        return candidate

    @property
    def history(self) -> list[Intervention]:
        """Recent intervention history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Reset the intervention history."""
        self._history.clear()

    # ── Internal helpers ──────────────────────────────────────────────

    def _random_intervention(self) -> Intervention:
        return Intervention(
            model=random.choice(self._config.models),
            temperature=random.choice(self._config.temperatures),
            skill_prefix=random.choice(self._config.skill_prefixes),
            method=random.choice(self._config.methods),
        )

    def _ensure_diversity(self, candidate: Intervention, last: Intervention) -> Intervention:
        """Mutate candidate axes until it differs from ``last`` on enough axes."""
        axes_changed = self._count_axes_changed(candidate, last)
        attempts = 0
        while axes_changed < self._min_axes_changed and attempts < 20:
            # Randomly change one axis at a time
            axis = random.choice(["model", "temperature", "skill_prefix", "method"])
            if axis == "model":
                candidate.model = random.choice(self._config.models)
            elif axis == "temperature":
                candidate.temperature = random.choice(self._config.temperatures)
            elif axis == "skill_prefix":
                candidate.skill_prefix = random.choice(self._config.skill_prefixes)
            elif axis == "method":
                candidate.method = random.choice(self._config.methods)
            axes_changed = self._count_axes_changed(candidate, last)
            attempts += 1
        return candidate

    @staticmethod
    def _count_axes_changed(a: Intervention, b: Intervention) -> int:
        """Count how many axes differ between two interventions."""
        count = 0
        if a.model != b.model:
            count += 1
        if abs(a.temperature - b.temperature) > 0.05:  # tolerance for floats
            count += 1
        if a.skill_prefix != b.skill_prefix:
            count += 1
        if a.method != b.method:
            count += 1
        return count
