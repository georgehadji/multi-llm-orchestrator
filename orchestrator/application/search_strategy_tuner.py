"""
SearchStrategyTuner — Level 1.5 Search Parameter Adjustment
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads ``SearchTrace`` from the ``TraceAnalyzer`` and emits ``SearchConfigUpdate``
objects that adjust quality thresholds, max-attempt limits, Verbalized Sampling
flags, and ARA eligibility based on detected stagnation.

The tuner persists its history to SQLite and can freeze/unfreeze individual
parameters to prevent oscillation.

Usage:
    tuner = SearchStrategyTuner()
    trace = await analyzer.analyze(project_ids=[...])
    update = await tuner.evaluate(trace)
    if update.has_changes:
        await tuner.apply(update, stage=my_self_consistency_stage)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .trace_analyzer import SearchTrace

logger = logging.getLogger("orchestrator.bilevel.strategy_tuner")


# ── Configuration data structures ─────────────────────────────────────────


@dataclass
class SearchConfig:
    """Current search parameter values.

    Defaults match the production values in ``SelfConsistencyStage``
    and Verbalized Sampling flags.
    """

    quality_threshold: float = 0.7
    max_attempts: int = 3
    vs_retry_escape: bool = False
    vs_test_generation: bool = False
    ara_eligibility: bool = True  # whether ARA execution strategy is eligible


@dataclass
class SearchConfigUpdate:
    """A set of parameter changes produced by the tuner."""

    quality_threshold: float | None = None
    max_attempts: int | None = None
    vs_retry_escape: bool | None = None
    vs_test_generation: bool | None = None
    ara_eligibility: bool | None = None
    reason: str = ""

    @property
    def has_changes(self) -> bool:
        """True if at least one field is non-None."""
        return any(
            v is not None
            for v in [
                self.quality_threshold,
                self.max_attempts,
                self.vs_retry_escape,
                self.vs_test_generation,
                self.ara_eligibility,
            ]
        )


# ── Tuner ─────────────────────────────────────────────────────────────────


class SearchStrategyTuner:
    """Adjust search parameters based on trace analysis.

    Args:
        config: Initial search configuration. Defaults to production values.
        db_path: Optional SQLite path for persisting tuning history.
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
        db_path: str | None = None,
    ) -> None:
        self._config = config or SearchConfig()
        self._db_path = db_path
        self._frozen_params: set[str] = set()  # parameter names locked against changes
        self._history: list[dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────

    async def evaluate(self, trace: SearchTrace) -> SearchConfigUpdate:
        """Analyze a ``SearchTrace`` and produce a configuration update.

        The tuner applies heuristics based on stagnation, repetition,
        and cost-effectiveness:

        - If stagnation > 50%: lower quality_threshold, increase max_attempts
        - If repetition > 30%: enable vs_retry_escape
        - If cost-effectiveness is poor: disable ara_eligibility, enable vs flags
        """
        updates: list[str] = []
        quality_threshold: float | None = None
        max_attempts: int | None = None
        vs_retry_escape: bool | None = None
        vs_test_generation: bool | None = None
        ara_eligibility: bool | None = None

        # Heuristic 1: Stagnation -> lower threshold, more attempts
        if trace.stagnation_score > 0.5 and "quality_threshold" not in self._frozen_params:
            quality_threshold = round(max(0.3, self._config.quality_threshold - 0.1), 2)
            updates.append(
                f"stagnation {trace.stagnation_score:.0%} -> threshold {quality_threshold}"
            )

        if trace.stagnation_score > 0.5 and "max_attempts" not in self._frozen_params:
            max_attempts = min(5, self._config.max_attempts + 1)
            updates.append(f"stagnation -> max_attempts {max_attempts}")

        # Heuristic 2: High repetition -> enable VS retry escape
        if trace.repetition_rate > 0.3 and "vs_retry_escape" not in self._frozen_params:
            vs_retry_escape = True
            updates.append(f"repetition {trace.repetition_rate:.0%} -> vs_retry_escape=True")

        # Heuristic 3: Low cost-effectiveness -> try VS test gen, limit ARA
        if trace.avg_cost_per_point > 0.5 and "ara_eligibility" not in self._frozen_params:
            ara_eligibility = False
            vs_test_generation = True
            updates.append(f"cost ${trace.avg_cost_per_point:.2f}/pt -> ara=False, vs_test=True")

        reason = "; ".join(updates) if updates else "no changes needed"

        return SearchConfigUpdate(
            quality_threshold=quality_threshold,
            max_attempts=max_attempts,
            vs_retry_escape=vs_retry_escape,
            vs_test_generation=vs_test_generation,
            ara_eligibility=ara_eligibility,
            reason=reason,
        )

    async def apply(
        self,
        update: SearchConfigUpdate,
        stage: Any = None,
    ) -> None:
        """Apply the update to the active configuration and optionally to a stage.

        Args:
            update: The configuration changes to apply.
            stage: Optional stage object (e.g., ``SelfConsistencyStage``)
                   whose parameters should be updated in-place.
        """
        if update.quality_threshold is not None:
            self._config.quality_threshold = update.quality_threshold
        if update.max_attempts is not None:
            self._config.max_attempts = update.max_attempts
        if update.vs_retry_escape is not None:
            self._config.vs_retry_escape = update.vs_retry_escape
        if update.vs_test_generation is not None:
            self._config.vs_test_generation = update.vs_test_generation
        if update.ara_eligibility is not None:
            self._config.ara_eligibility = update.ara_eligibility

        # Persist to history
        self._history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "update": {
                    "quality_threshold": update.quality_threshold,
                    "max_attempts": update.max_attempts,
                    "vs_retry_escape": update.vs_retry_escape,
                    "vs_test_generation": update.vs_test_generation,
                    "ara_eligibility": update.ara_eligibility,
                },
                "reason": update.reason,
            }
        )

        # Apply to stage if provided
        if stage is not None:
            if hasattr(stage, "max_attempts") and update.max_attempts is not None:
                stage.max_attempts = update.max_attempts
            if hasattr(stage, "quality_threshold") and update.quality_threshold is not None:
                stage.quality_threshold = update.quality_threshold

        if update.has_changes:
            logger.info(
                "SearchStrategyTuner applied: %s",
                update.reason or "unknown reason",
            )

    def freeze(self, param_name: str) -> None:
        """Lock a parameter so ``evaluate()`` will not change it."""
        self._frozen_params.add(param_name)

    def unfreeze(self, param_name: str) -> None:
        """Unlock a previously frozen parameter."""
        self._frozen_params.discard(param_name)

    @property
    def config(self) -> SearchConfig:
        """Current active configuration."""
        return self._config

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only tuning history."""
        return list(self._history)
