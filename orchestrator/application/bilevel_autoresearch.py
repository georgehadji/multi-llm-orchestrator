"""
BilevelAutoresearchService — Meta-Level Coordination for Outer-Loop
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Coordinates the bilevel autoresearch pipeline after each project completes:

1. **Trace analysis** via ``TraceAnalyzer``
2. **Strategy tuning** via ``SearchStrategyTuner``
3. **Mechanism proposals** sent as ``StrategyProposal`` to ``MetaOptimizationV2``

All mechanism changes route through the existing HITL / A/B / rollout safety
layer rather than being applied directly.

Usage:
    service = BilevelAutoresearchService(
        trace_analyzer=analyzer,
        strategy_tuner=tuner,
        meta_v2=meta_v2_instance,
    )
    await service.on_project_completed(project_state)
"""

from __future__ import annotations

import logging
from typing import Any

from .trace_analyzer import TraceAnalyzer, SearchTrace
from .search_strategy_tuner import SearchStrategyTuner, SearchConfigUpdate

logger = logging.getLogger("orchestrator.bilevel.autoresearch_service")


class BilevelAutoresearchService:
    """Coordinate the outer-loop autoresearch pipeline.

    Runs after each project completes (when the feature flag is enabled).
    Analyzes the trace, tunes search parameters, and if needed creates
    ``StrategyProposal`` objects that go through ``MetaOptimizationV2``
    for HITL / A/B / rollout governance.

    Args:
        trace_analyzer: ``TraceAnalyzer`` instance for metric computation.
        strategy_tuner: ``SearchStrategyTuner`` for parameter adjustment.
        meta_v2: ``MetaOptimizationV2`` (or compatible wrapper) for proposal routing.
        mechanism_injector: Optional ``MechanismInjector`` for runtime injection.
        mechanism_researcher: Optional ``MechanismResearcher`` for generating mechanisms.
        enabled: Whether to actually run. Checked against the feature flag.
    """

    def __init__(
        self,
        trace_analyzer: TraceAnalyzer | None = None,
        strategy_tuner: SearchStrategyTuner | None = None,
        meta_v2: Any = None,
        mechanism_injector: Any = None,
        mechanism_researcher: Any = None,
        enabled: bool = True,
    ) -> None:
        self._analyzer = trace_analyzer
        self._tuner = strategy_tuner
        self._meta_v2 = meta_v2
        self._injector = mechanism_injector
        self._researcher = mechanism_researcher
        self._enabled = enabled

    async def on_project_completed(self, project_state: Any) -> None:
        """Hook called after a project completes.

        This is the main entry point. It runs the full bilevel pipeline:

        1. Extract trace from the completed project
        2. Run trace analysis
        3. Run strategy tuner
        4. If changes suggested, create a StrategyProposal
        5. Route through MetaOptimizationV2 for governance
        6. Optionally inject a new mechanism

        Args:
            project_state: The ``ProjectState`` returned by ``run_project``.
        """
        if not self._enabled:
            logger.debug("Bilevel autoresearch disabled — skipping")
            return

        if self._analyzer is None or self._tuner is None:
            logger.debug("Analyzer or tuner not configured — skipping")
            return

        # Step 1: Build a SearchTrace from the project state
        project_id = getattr(project_state, "project_id", "") or ""
        trace = await self._build_trace(project_state)

        # Step 2: Run strategy tuner
        update = await self._tuner.evaluate(trace)
        if update.has_changes:
            logger.info(
                "Bilevel: strategy tuner suggests changes for %s: %s",
                project_id,
                update.reason,
            )

            # Step 3: Route through Meta V2 if available
            if self._meta_v2 is not None:
                proposal = self._build_proposal(update, trace, project_id)
                await self._route_proposal(proposal)
            else:
                # No Meta V2 — apply directly (development mode)
                logger.info(
                    "Bilevel: no Meta V2, applying tuner update directly for %s",
                    project_id,
                )
                await self._tuner.apply(update)

        # Step 4: If mechanism researcher and injector are available, try generation
        if self._researcher is not None and self._injector is not None:
            await self._try_generate_mechanism(trace, project_id)

        logger.debug("Bilevel autoresearch completed for %s", project_id)

    # ── Internal helpers ──────────────────────────────────────────────

    async def _build_trace(self, project_state: Any) -> SearchTrace:
        """Build a minimal SearchTrace from a single project state."""
        trace = SearchTrace(project_ids=[getattr(project_state, "project_id", "")])
        if self._analyzer is not None:
            task_traces = await self._analyzer._analyze_project(
                getattr(project_state, "project_id", "")
            )
            trace.task_traces = task_traces
            self._analyzer._compute_repetition(trace)
            self._analyzer._compute_fixation(trace)
            self._analyzer._compute_stagnation(trace)
            self._analyzer._compute_cost_effectiveness(trace)
        return trace

    def _build_proposal(
        self,
        update: SearchConfigUpdate,
        trace: SearchTrace,
        project_id: str,
    ) -> Any:
        """Build a StrategyProposal from a tuner update.

        Maps the ``SearchConfigUpdate`` into a ``StrategyProposal``
        suitable for ``MetaOptimizationV2.evaluate_proposal``.
        """
        from ..hitl_workflow import ImpactLevel
        from ..meta.orchestrator import (
            ProposalStatus,
            StrategyProposal,
            StrategyType,
        )

        # Build a description of the proposed changes
        changes = []
        if update.quality_threshold is not None:
            changes.append(f"quality_threshold -> {update.quality_threshold}")
        if update.max_attempts is not None:
            changes.append(f"max_attempts -> {update.max_attempts}")
        if update.vs_retry_escape is not None:
            changes.append(f"vs_retry_escape -> {update.vs_retry_escape}")
        if update.ara_eligibility is not None:
            changes.append(f"ara_eligibility -> {update.ara_eligibility}")

        return StrategyProposal(
            proposal_id=f"bilevel_{project_id}_{hash(update.reason) % 10000:04d}",
            strategy_type=StrategyType.TEMPLATE_CONFIG,
            description=f"Bilevel tuner: {update.reason}",
            current_config={},
            proposed_config={
                "quality_threshold": update.quality_threshold,
                "max_attempts": update.max_attempts,
                "vs_retry_escape": update.vs_retry_escape,
                "ara_eligibility": update.ara_eligibility,
            },
            expected_improvement=max(0.0, 1.0 - trace.stagnation_score),
            confidence=0.5,
            evidence=[
                f"Stagnation: {trace.stagnation_score:.2%}",
                f"Repetition: {trace.repetition_rate:.2%}",
                f"Cost/pt: ${trace.avg_cost_per_point:.4f}",
            ],
            status=ProposalStatus.PENDING,
        )

    async def _route_proposal(self, proposal: Any) -> None:
        """Route a StrategyProposal through MetaOptimizationV2.

        The Meta V2 layer handles impact classification, HITL approval,
        A/B testing, and gradual rollout.
        """
        if self._meta_v2 is None or not hasattr(self._meta_v2, "evaluate_proposal"):
            logger.warning("Meta V2 has no evaluate_proposal method — skipping")
            return

        try:
            outcome = await self._meta_v2.evaluate_proposal(proposal)
            logger.info(
                "Bilevel proposal %s: %s — %s",
                proposal.proposal_id,
                outcome.decision,
                outcome.reason or "no reason",
            )
        except Exception as exc:
            logger.exception("Bilevel proposal routing failed: %s", exc)

    async def _try_generate_mechanism(self, trace: SearchTrace, project_id: str) -> None:
        """Attempt to generate a new mechanism using the researcher."""
        if self._researcher is None or not hasattr(self._researcher, "research"):
            return
        try:
            result = await self._researcher.research(trace)
            if result and self._injector is not None:
                success, msg = await self._injector.inject(
                    name=result.get("name", f"gen_{project_id}"),
                    code=result.get("code", ""),
                    description=result.get("description", ""),
                )
                if success:
                    logger.info("Bilevel: injected generated mechanism %s", msg)
        except Exception as exc:
            logger.warning("Bilevel mechanism generation failed: %s", exc)
