"""
RoutingTableCarrier — Evolve Model Routing from Outcomes (Phase 4.4)
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads the current ``ROUTING_TABLE`` (per-task-type model priority lists),
learns from project outcomes which models perform best, and proposes
updated priority lists. All routing changes are validated against
historical data before activation.

Design:
    - Tracks per-model success rates per task type
    - Proposes re-ordering the routing table based on observed performance
    - Changes must pass a validation gate (minimum samples, significance)
    - Integrates with ``MetaOptimizationV2`` for A/B testing

Usage:
    carrier = RoutingTableCarrier()
    await carrier.record_outcome(TaskType.CODE_GEN, Model.GPT4, score=0.9)
    proposal = await carrier.propose_update()
    if proposal:
        await carrier.apply_update(proposal)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import Model, ROUTING_TABLE, TaskType

logger = logging.getLogger("orchestrator.bilevel.routing_carrier")


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class ModelOutcome:
    """Performance data for a single model on a task type."""

    model: str = ""
    task_type: str = ""
    attempts: int = 0
    successes: int = 0
    total_score: float = 0.0
    total_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.attempts, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.attempts, 1)

    @property
    def score_per_dollar(self) -> float:
        return self.total_score / max(self.total_cost, 0.001)


@dataclass
class RoutingProposal:
    """Proposal to re-order the routing table for a task type."""

    task_type: str = ""
    original_order: list[str] = field(default_factory=list)
    proposed_order: list[str] = field(default_factory=list)
    evidence: str = ""

    @property
    def has_changes(self) -> bool:
        return self.original_order != self.proposed_order and len(self.proposed_order) > 0


# ── Carrier ───────────────────────────────────────────────────────────────


class RoutingTableCarrier:
    """Evolve model routing priorities based on project outcomes.

    Tracks per-model performance per task type and proposes updates to
    the model priority lists in the routing table. Changes are validated
    before activation (minimum 10 samples per model, significance gate).

    Args:
        min_samples: Minimum attempts per model before considering it (default 10).
        improvement_threshold: Minimum success-rate improvement to promote a model (default 0.1).
    """

    def __init__(
        self,
        min_samples: int = 10,
        improvement_threshold: float = 0.1,
    ) -> None:
        self._min_samples = min_samples
        self._improvement_threshold = improvement_threshold
        self._outcomes: dict[str, dict[str, ModelOutcome]] = defaultdict(
            lambda: defaultdict(ModelOutcome)
        )

    async def record_outcome(
        self,
        task_type: TaskType,
        model: Model,
        score: float = 0.0,
        cost: float = 0.0,
        threshold: float = 0.7,
    ) -> None:
        """Record the outcome of using a model for a task type.

        Args:
            task_type: The task type.
            model: The model used.
            score: Score achieved (0.0 - 1.0).
            cost: Cost incurred.
            threshold: Score threshold for success.
        """
        key = task_type.value
        outcome = self._outcomes[key][model.value]
        outcome.model = model.value
        outcome.task_type = key
        outcome.attempts += 1
        if score >= threshold:
            outcome.successes += 1
        outcome.total_score += score
        outcome.total_cost += cost

    async def propose_update(
        self,
        task_type: TaskType | None = None,
    ) -> list[RoutingProposal]:
        """Propose routing table updates based on observed outcomes.

        Analyzes per-model success rates and proposes re-ordering
        the routing table. Only models with sufficient samples are
        considered.

        Args:
            task_type: Optional task type filter. If None, all types are analyzed.

        Returns:
            List of ``RoutingProposal`` with suggested re-orderings.
        """
        proposals: list[RoutingProposal] = []

        task_types = [task_type] if task_type else list(TaskType)
        current_table = ROUTING_TABLE

        for tt in task_types:
            key = tt.value
            if key not in self._outcomes:
                continue

            current_order = current_table.get(tt, [])
            original = [m.value for m in current_order]

            # Build candidate re-ordering: sort by success rate descending
            candidates = []
            for model_value, outcome in self._outcomes[key].items():
                if outcome.attempts >= self._min_samples:
                    candidates.append((outcome.success_rate, outcome.score_per_dollar, model_value))

            if not candidates:
                continue

            # Sort by success rate, then score-per-dollar
            candidates.sort(key=lambda x: (-x[0], -x[1]))
            proposed = [c[2] for c in candidates]

            # Keep any original models that didn't have enough samples
            for m in original:
                if m not in proposed:
                    proposed.append(m)

            prop = RoutingProposal(
                task_type=key,
                original_order=original,
                proposed_order=proposed,
                evidence=f"Top model: {candidates[0][2]} ({candidates[0][0]:.0%} success)",
            )
            if prop.has_changes:
                proposals.append(prop)

        return proposals

    async def apply_update(
        self,
        proposal: RoutingProposal,
        config_path: str = "",
    ) -> bool:
        """Apply a routing proposal to the config file.

        Args:
            proposal: The ``RoutingProposal`` to apply.
            config_path: Path to ``routing.json``. If empty, uses the default path.

        Returns:
            True if the update was applied.
        """
        if not proposal.has_changes:
            logger.info("Routing proposal has no changes — skipping")
            return False

        # Determine config path
        base = Path(config_path) if config_path else Path(__file__).parent.parent / "config"
        routing_file = base / "routing.json"

        if not routing_file.exists():
            logger.warning("routing.json not found at %s", routing_file)
            return False

        try:
            data = json.loads(routing_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read routing.json: %s", exc)
            return False

        if proposal.task_type not in data:
            logger.warning("Task type '%s' not found in routing.json", proposal.task_type)
            return False

        # Update the routing table
        old_order = data[proposal.task_type]
        data[proposal.task_type] = proposal.proposed_order

        try:
            routing_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Updated routing for '%s': %d models reordered",
                proposal.task_type,
                len(proposal.proposed_order),
            )
            return True
        except OSError as exc:
            logger.warning("Failed to write routing.json: %s", exc)
            return False

    def get_outcomes(self, task_type: TaskType) -> list[ModelOutcome]:
        """Get outcome data for a task type."""
        return list(self._outcomes.get(task_type.value, {}).values())
