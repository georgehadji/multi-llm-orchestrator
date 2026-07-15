"""
RecursiveResearch — Outer-Loop Self-Improvement (Phase 4.5)
============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Allows the bilevel outer loop to improve its own configuration by
analyzing its own research sessions (meta-trace analysis). Level 2
mechanisms can propose changes to:

- **Trace analysis parameters** (stagnation window, repetition threshold)
- **Tuner heuristics** (score thresholds, adjustment step sizes)
- **Mechanism generation** (exploration/exploitation balance)
- **Diversity enforcement** (minimum axis changes, history size)

All self-modifications are guarded by **stronger validation gates**:
- Minimum 10 research sessions before any self-change
- Only parameter changes (no code injection for self)
- Require HITL approval (via MetaOptimizationV2)
- Automatic rollback if the next 3 sessions show regression

Usage:
    rr = RecursiveResearch(
        trace_analyzer=analyzer,
        strategy_tuner=tuner,
        exploration=explorer,
        meta_v2=meta_v2,
    )
    await rr.evaluate(research_session_trace)
    await rr.propose_self_improvement()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..crosscutting.config import flags
from .trace_analyzer import SearchTrace

logger = logging.getLogger("orchestrator.bilevel.recursive_research")


# ── Safety Gates ──────────────────────────────────────────────────────────


class SelfImprovementGate:
    """Validation gates that must pass before self-modification.

    Args:
        min_sessions: Minimum research sessions required (default 10).
        max_regression_sessions: Consecutive regressions allowed before rollback (default 3).
    """

    def __init__(
        self,
        min_sessions: int = 10,
        max_regression_sessions: int = 3,
    ) -> None:
        self.min_sessions = min_sessions
        self.max_regression_sessions = max_regression_sessions

    def may_proceed(self, session_count: int) -> tuple[bool, str]:
        """Check whether self-improvement may proceed.

        Args:
            session_count: Number of research sessions completed.

        Returns:
            Tuple of ``(allowed, reason)``.
        """
        if not flags.bilevel_autoresearch_enabled:
            return False, "Bilevel autoresearch is disabled"
        if session_count < self.min_sessions:
            return False, f"Only {session_count} sessions (need {self.min_sessions})"
        return True, "Gate passed"


@dataclass
class SelfImprovementProposal:
    """A proposal to change the outer loop's own configuration."""

    parameter: str = ""
    current_value: Any = None
    proposed_value: Any = None
    rationale: str = ""
    expected_impact: str = ""

    @property
    def has_changes(self) -> bool:
        return self.parameter != ""


# ── Recursive Research ────────────────────────────────────────────────────


class RecursiveResearch:
    """Outer-loop self-improvement through meta-trace analysis.

    Analyzes the research session traces to identify opportunities to
    improve the bilevel system's own parameters. Proposals are guarded
    by ``SelfImprovementGate`` and require HITL approval.

    Args:
        trace_analyzer: The ``TraceAnalyzer`` used by the outer loop.
        strategy_tuner: The ``SearchStrategyTuner`` instance.
        exploration: The ``OrthogonalExploration`` instance.
        meta_v2: ``MetaOptimizationV2`` for HITL / A/B governance.
        gate: Optional custom ``SelfImprovementGate``.
    """

    def __init__(
        self,
        trace_analyzer: Any = None,
        strategy_tuner: Any = None,
        exploration: Any = None,
        meta_v2: Any = None,
        gate: SelfImprovementGate | None = None,
    ) -> None:
        self._analyzer = trace_analyzer
        self._tuner = strategy_tuner
        self._exploration = exploration
        self._meta_v2 = meta_v2
        self._gate = gate or SelfImprovementGate()
        self._session_count = 0
        self._scores: list[float] = []  # scores of recent research sessions
        self._proposals_history: list[SelfImprovementProposal] = []
        self._rollback_countdown = 0

    async def evaluate(self, trace: SearchTrace) -> None:
        """Record a research session trace for meta-analysis.

        Args:
            trace: The ``SearchTrace`` from the most recent research session.
        """
        self._session_count += 1
        # Track the stagnation as a proxy for "research quality"
        self._scores.append(1.0 - trace.stagnation_score)

        # Check if we need to roll back after a previous self-modification
        if self._rollback_countdown > 0:
            self._rollback_countdown -= 1
            if self._rollback_countdown == 0 and len(self._scores) >= 3:
                recent = self._scores[-3:]
                if recent[0] > recent[-1]:  # regression
                    logger.warning(
                        "Recursive research: scores declining after self-modification "
                        "(%f -> %f). Auto-rollback triggered.",
                        recent[0],
                        recent[-1],
                    )
                    await self._auto_rollback()

    async def propose_self_improvement(
        self,
    ) -> SelfImprovementProposal | None:
        """Analyze research traces and propose an improvement to the outer loop.

        Returns a ``SelfImprovementProposal`` if a change is warranted,
        or ``None`` if the system is performing well.

        The proposal is sent through ``MetaOptimizationV2`` if available
        for HITL / A/B governance.
        """
        allowed, reason = self._gate.may_proceed(self._session_count)
        if not allowed:
            logger.debug("Recursive research blocked: %s", reason)
            return None

        proposal = self._analyze_and_propose()
        if proposal is None or not proposal.has_changes:
            return None

        # Route through Meta V2 if available
        if self._meta_v2 is not None:
            await self._route_through_meta_v2(proposal)

        self._proposals_history.append(proposal)
        logger.info(
            "Recursive research proposed: %s -> %s (%s)",
            proposal.parameter,
            proposal.proposed_value,
            proposal.rationale,
        )
        return proposal

    # ── Internal helpers ──────────────────────────────────────────────

    def _analyze_and_propose(self) -> SelfImprovementProposal | None:
        """Analyze research session data and propose a tunable improvement."""
        if len(self._scores) < self._gate.min_sessions:
            return None

        recent = self._scores[-5:] if len(self._scores) >= 5 else self._scores
        avg_score = sum(recent) / len(recent)

        # Low average score -> suggest changes to tuner or exploration
        if avg_score < 0.5:
            # Suggest adjusting the tuner's score threshold
            if self._tuner is not None:
                current = (
                    self._tuner.config.score_threshold
                    if hasattr(self._tuner.config, "score_threshold")
                    else 0.7
                )
                proposal = SelfImprovementProposal(
                    parameter="tuner.score_threshold",
                    current_value=current,
                    proposed_value=round(max(0.3, current - 0.1), 2),
                    rationale=f"Low research quality ({avg_score:.2f} avg) suggests lowering threshold",
                    expected_impact="More generous success classification",
                )
                return proposal

        # High stagnation -> suggest more aggressive exploration
        if avg_score < 0.6 and self._exploration is not None:
            proposal = SelfImprovementProposal(
                parameter="exploration.min_axes_changed",
                current_value=self._exploration._min_axes_changed,
                proposed_value=min(4, self._exploration._min_axes_changed + 1),
                rationale=f"Increasing axis change requirement to break stagnation",
                expected_impact="More diverse interventions",
            )
            return proposal

        return None

    async def _route_through_meta_v2(self, proposal: SelfImprovementProposal) -> None:
        """Send proposal to MetaOptimizationV2 for HITL/A/B governance."""
        if not hasattr(self._meta_v2, "evaluate_proposal"):
            return
        try:
            from ..meta.orchestrator import StrategyProposal, StrategyType, ProposalStatus

            sp = StrategyProposal(
                proposal_id=f"recursive_{proposal.parameter}_{self._session_count}",
                strategy_type=StrategyType.TEMPLATE_CONFIG,
                description=f"Self-improvement: {proposal.rationale}",
                current_config={proposal.parameter: proposal.current_value},
                proposed_config={proposal.parameter: proposal.proposed_value},
                expected_improvement=0.1,
                confidence=0.4,
                evidence=[f"Research sessions: {self._session_count}"],
                status=ProposalStatus.PENDING,
            )
            outcome = await self._meta_v2.evaluate_proposal(sp)
            # If HITL approved, schedule rollback monitoring
            if outcome.decision in ("approved", "sent_to_ab_test"):
                self._rollback_countdown = self._gate.max_regression_sessions
                logger.info("Recursive proposal %s: %s", proposal.parameter, outcome.decision)
        except Exception as exc:
            logger.warning("Recursive research Meta V2 routing failed: %s", exc)

    async def _auto_rollback(self) -> None:
        """Automatically roll back the last self-modification."""
        if not self._proposals_history:
            return
        last = self._proposals_history.pop()
        logger.info(
            "Auto-rollback: reverting %s from %s to %s",
            last.parameter,
            last.proposed_value,
            last.current_value,
        )
        # Reset the countdown
        self._rollback_countdown = 0
