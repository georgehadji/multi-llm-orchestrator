"""
Meta-Optimization V2 Integration
=================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Unified integration layer for meta-optimization with:
- A/B Testing
- Human-in-the-Loop (HITL)
- Gradual Rollout

This integrates all Phase 1 & 2 features into a cohesive system.

USAGE:
    from orchestrator.meta_v2_integration import MetaOptimizationV2, MetaV2Config

    config = MetaV2Config(
        ab_testing_enabled=True,
        hitl_enabled=True,
        rollout_enabled=True,
    )

    meta_v2 = MetaOptimizationV2(orchestrator, archive, config)

    # After project completion
    await meta_v2.record_project_completion(trajectory)

    # Periodic optimization
    proposals = await meta_v2.maybe_optimize()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .ab_testing import ABTestingEngine, Recommendation
from .gradual_rollout import (
    GradualRolloutManager,
    RolloutConfig,
)
from .hitl_workflow import (
    ApprovalConfig,
    ApprovalStatus,
    HITLWorkflow,
    ImpactLevel,
)
from .meta_orchestrator import (
    ExecutionArchive,
    MetaOptimizer,
    ProjectTrajectory,
    ProposalStatus,
    StrategyProposal,
)

logger = logging.getLogger("orchestrator.meta_v2")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class MetaV2Config:
    """Configuration for Meta-Optimization V2."""

    # A/B Testing
    ab_testing_enabled: bool = True
    ab_traffic_split: float = 0.1
    ab_min_samples: int = 30
    ab_significance_level: float = 0.05

    # HITL
    hitl_enabled: bool = True
    hitl_auto_approve_low_risk: bool = True
    hitl_approval_timeout_hours: float = 72.0

    # Gradual Rollout
    rollout_enabled: bool = True
    rollout_auto_rollback: bool = True

    # General
    storage_path: Path | None = None
    min_executions_for_optimization: int = 50
    max_proposals_per_cycle: int = 3


class ProposalDecision(str, Enum):
    """Decision for a proposal."""
    AUTO_APPROVED = "auto_approved"
    SENT_TO_HITL = "sent_to_hitl"
    SENT_TO_AB_TEST = "sent_to_ab_test"
    SENT_TO_ROLLOUT = "sent_to_rollout"
    REJECTED = "rejected"


@dataclass
class ProposalOutcome:
    """Outcome of proposal evaluation."""
    proposal: StrategyProposal
    decision: ProposalDecision
    hitl_request_id: str | None = None
    experiment_id: str | None = None
    rollout_id: str | None = None
    reason: str | None = None


# ─────────────────────────────────────────────
# Meta-Optimization V2
# ─────────────────────────────────────────────

class MetaOptimizationV2:
    """
    Meta-Optimization V2 with A/B Testing, HITL, and Gradual Rollout.

    Provides a complete pipeline for safe strategy optimization.
    """

    def __init__(
        self,
        orchestrator: Any,
        archive: ExecutionArchive,
        config: MetaV2Config | None = None,
    ):
        self.orchestrator = orchestrator
        self.archive = archive
        self.config = config or MetaV2Config()

        # Initialize components
        self.optimizer = MetaOptimizer(archive)

        self.ab_testing: ABTestingEngine | None = None
        self.hitl: HITLWorkflow | None = None
        self.rollout: GradualRolloutManager | None = None

        if self.config.ab_testing_enabled:
            self.ab_testing = ABTestingEngine(
                archive,
                storage_path=self._get_storage_path() / "ab_testing",
            )

        if self.config.hitl_enabled:
            approval_config = ApprovalConfig(
                auto_approve_low_risk=self.config.hitl_auto_approve_low_risk,
                approval_timeout_hours=self.config.hitl_approval_timeout_hours,
                storage_path=self._get_storage_path(),
            )
            self.hitl = HITLWorkflow(approval_config)

        if self.config.rollout_enabled:
            rollout_config = RolloutConfig(
                auto_rollback_enabled=self.config.rollout_auto_rollback,
                storage_path=self._get_storage_path() / "rollouts",
            )
            self.rollout = GradualRolloutManager(archive, rollout_config)

        self._lock = asyncio.Lock()
        self._execution_count = 0

    def _get_storage_path(self) -> Path:
        """Get storage path."""
        return self.config.storage_path or (
            Path.home() / ".orchestrator_cache" / "meta_v2"
        )

    async def record_project_completion(self, trajectory: ProjectTrajectory):
        """
        Record a completed project for meta-optimization.

        Also records outcomes for active experiments and rollouts.
        """
        async with self._lock:
            self._execution_count += 1

            # Record to archive
            self.archive.store(trajectory)

            # Record outcomes for active experiments
            if self.ab_testing:
                await self._record_experiment_outcomes(trajectory)

            # Record outcomes for active rollouts
            if self.rollout:
                await self._record_rollout_outcomes(trajectory)

    async def _record_experiment_outcomes(self, trajectory: ProjectTrajectory):
        """Record outcomes for active experiments."""
        if not self.ab_testing:
            return

        active_experiments = await self.ab_testing.get_active_experiments()

        for experiment in active_experiments:
            # Determine variant for this project
            variant = await self.ab_testing.route_execution(trajectory.project_id)

            # Record outcome for each task
            for task_record in trajectory.task_records:
                await self.ab_testing.record_outcome(
                    experiment_id=experiment.experiment_id,
                    variant=variant,
                    project_id=trajectory.project_id,
                    success=task_record.success,
                    score=task_record.score,
                    cost_usd=task_record.cost_usd,
                    latency_ms=task_record.latency_ms,
                )

    async def _record_rollout_outcomes(self, trajectory: ProjectTrajectory):
        """Record outcomes for active rollouts."""
        if not self.rollout:
            return

        active_rollouts = await self.rollout.get_active_rollouts()

        for rollout in active_rollouts:
            # Record outcome for each task
            for task_record in trajectory.task_records:
                await self.rollout.record_execution(
                    rollout_id=rollout.rollout_id,
                    success=task_record.success,
                    score=task_record.score,
                    cost_usd=task_record.cost_usd,
                    latency_ms=task_record.latency_ms,
                    project_id=trajectory.project_id,
                )

    async def maybe_optimize(self) -> list[ProposalOutcome]:
        """
        Periodically analyze and propose optimizations.

        Returns list of proposal outcomes.
        """
        async with self._lock:
            # Check minimum executions
            if self.archive.total_executions < self.config.min_executions_for_optimization:
                logger.info(
                    f"Insufficient executions for optimization: "
                    f"{self.archive.total_executions} < {self.config.min_executions_for_optimization}"
                )
                return []

            # Generate proposals
            proposals = await self.optimizer.analyze_and_propose()

            outcomes = []
            for proposal in proposals[:self.config.max_proposals_per_cycle]:
                outcome = await self._evaluate_proposal(proposal)
                outcomes.append(outcome)

            # Check experiment results
            if self.ab_testing:
                await self._check_experiments()

            # Check rollout progress
            if self.rollout:
                await self._check_rollouts()

            return outcomes

    async def _evaluate_proposal(self, proposal: StrategyProposal) -> ProposalOutcome:
        """Evaluate a single proposal through the pipeline."""

        # Determine impact level
        impact_level = self._determine_impact_level(proposal)

        # Low impact + high confidence → Auto-approve
        if (impact_level == ImpactLevel.LOW and
            proposal.confidence >= 0.9 and
            self.config.hitl_enabled and
            self.hitl.config.auto_approve_low_risk):

            proposal.status = ProposalStatus.APPLIED
            logger.info(f"Auto-approved low-impact proposal: {proposal.proposal_id}")

            return ProposalOutcome(
                proposal=proposal,
                decision=ProposalDecision.AUTO_APPROVED,
                reason="Low impact, high confidence",
            )

        # Medium/High impact → A/B Test
        if (impact_level in [ImpactLevel.MEDIUM, ImpactLevel.HIGH] and
            self.config.ab_testing_enabled and
            self.ab_testing):

            experiment = await self.ab_testing.create_experiment(
                proposal,
                traffic_split=self.config.ab_traffic_split,
                min_samples=self.config.ab_min_samples,
            )

            logger.info(
                f"Created A/B test for proposal {proposal.proposal_id}: "
                f"experiment {experiment.experiment_id}"
            )

            return ProposalOutcome(
                proposal=proposal,
                decision=ProposalDecision.SENT_TO_AB_TEST,
                experiment_id=experiment.experiment_id,
                reason=f"A/B test with {self.config.ab_traffic_split:.0%} traffic",
            )

        # Structural → HITL
        if (impact_level == ImpactLevel.STRUCTURAL and
            self.config.hitl_enabled and
            self.hitl):

            request = await self.hitl.submit_for_approval(proposal)

            if request.auto_approved:
                return ProposalOutcome(
                    proposal=proposal,
                    decision=ProposalDecision.AUTO_APPROVED,
                    reason=request.auto_approve_reason,
                )

            logger.info(
                f"Submitted structural proposal to HITL: "
                f"request {request.request_id}"
            )

            return ProposalOutcome(
                proposal=proposal,
                decision=ProposalDecision.SENT_TO_HITL,
                hitl_request_id=request.request_id,
                reason="Structural change requires approval",
            )

        # Default → Direct rollout
        if self.config.rollout_enabled and self.rollout:
            rollout = await self.rollout.start_rollout(proposal)

            logger.info(
                f"Started gradual rollout for proposal {proposal.proposal_id}: "
                f"rollout {rollout.rollout_id}"
            )

            return ProposalOutcome(
                proposal=proposal,
                decision=ProposalDecision.SENT_TO_ROLLOUT,
                rollout_id=rollout.rollout_id,
                reason="Standard rollout",
            )

        # Fallback → Reject
        return ProposalOutcome(
            proposal=proposal,
            decision=ProposalDecision.REJECTED,
            reason="No evaluation path available",
        )

    def _determine_impact_level(self, proposal: StrategyProposal) -> ImpactLevel:
        """Determine impact level based on proposal characteristics."""
        # Check description for keywords
        desc_lower = proposal.description.lower()

        if "structural" in desc_lower or "core" in desc_lower:
            return ImpactLevel.STRUCTURAL

        if proposal.strategy_type.value == "budget_allocation":
            factor = proposal.proposed_config.get("budget_factor", 1.0)
            current = proposal.current_config.get("budget_factor", 1.0)
            change = abs(factor - current) / max(current, 0.001)
            if change > 0.2:
                return ImpactLevel.HIGH
            return ImpactLevel.MEDIUM

        if proposal.strategy_type.value == "model_routing":
            if proposal.proposed_config.get("enabled") is False:
                return ImpactLevel.HIGH
            return ImpactLevel.MEDIUM

        return ImpactLevel.LOW

    async def _check_experiments(self):
        """Check and analyze completed experiments."""
        if not self.ab_testing:
            return

        active_experiments = await self.ab_testing.get_active_experiments()

        for experiment in active_experiments:
            result = await self.ab_testing.analyze_results(experiment.experiment_id)

            if result:
                if result.recommendation == Recommendation.ADOPT:
                    # Start rollout for winning variant
                    if self.rollout:
                        await self.rollout.start_rollout(experiment.proposal)
                        logger.info(
                            f"Experiment {experiment.experiment_id} adopted, "
                            f"starting rollout"
                        )
                else:
                    logger.info(
                        f"Experiment {experiment.experiment_id} rejected: "
                        f"{result.reasoning}"
                    )

    async def _check_rollouts(self):
        """Check and progress active rollouts."""
        if not self.rollout:
            return

        active_rollouts = await self.rollout.get_active_rollouts()

        for rollout in active_rollouts:
            decision = await self.rollout.check_stage_progress(rollout.rollout_id)

            if decision.decision == "advance":
                await self.rollout.advance_stage(rollout.rollout_id)
            elif decision.decision == "rollback":
                await self.rollout.trigger_rollback(rollout.rollout_id, decision.reason)

    async def check_hitl_status(self, request_id: str) -> ProposalOutcome | None:
        """
        Check status of a HITL request.

        Returns ProposalOutcome if approved, None if still pending.
        """
        if not self.hitl:
            return None

        request = await self.hitl.get_request(request_id)
        if not request:
            return None

        if request.status == ApprovalStatus.APPROVED:
            # Start rollout for approved proposal
            if self.rollout:
                rollout = await self.rollout.start_rollout(request.proposal)

                return ProposalOutcome(
                    proposal=request.proposal,
                    decision=ProposalDecision.SENT_TO_ROLLOUT,
                    rollout_id=rollout.rollout_id,
                    reason="HITL approved, starting rollout",
                )

        return None

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive meta-optimization status."""
        status = {
            "archive": {
                "total_projects": self.archive.total_projects,
                "total_executions": self.archive.total_executions,
            },
            "optimization": {
                "min_executions_required": self.config.min_executions_for_optimization,
                "ready_for_optimization": (
                    self.archive.total_executions >=
                    self.config.min_executions_for_optimization
                ),
            },
        }

        if self.ab_testing:
            status["ab_testing"] = self.ab_testing.get_experiment_stats()

        if self.hitl:
            status["hitl"] = self.hitl.get_stats()

        if self.rollout:
            status["rollout"] = self.rollout.get_rollout_stats()

        return status


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_meta_v2: MetaOptimizationV2 | None = None


def get_meta_optimization_v2() -> MetaOptimizationV2 | None:
    """Get global Meta-Optimization V2 instance."""
    return _meta_v2


def initialize_meta_optimization_v2(
    orchestrator: Any,
    archive: ExecutionArchive,
    config: MetaV2Config | None = None,
) -> MetaOptimizationV2:
    """Initialize global Meta-Optimization V2 instance."""
    global _meta_v2
    _meta_v2 = MetaOptimizationV2(orchestrator, archive, config)
    return _meta_v2


def reset_meta_optimization_v2() -> None:
    """Reset global Meta-Optimization V2 instance (for testing)."""
    global _meta_v2
    _meta_v2 = None
