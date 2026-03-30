"""
Meta-Optimization V2 Integration with Engine
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Minimal-invasive integration of Meta-Optimization V2 with Transfer Learning
into the main Orchestrator engine.

USAGE (in engine.py __init__):
    from .meta_integration import initialize_meta_optimization

    # After existing initialization
    self.meta_v2 = initialize_meta_optimization(
        self,
        self.state_mgr,
        enable_transfer_learning=True,
    )

USAGE (in engine.py run_project):
    # After project completion
    if self.meta_v2:
        await self.meta_v2.record_project_completion(state)
        await self.meta_v2.maybe_optimize()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("orchestrator.meta_integration")

if TYPE_CHECKING:
    from .engine import Orchestrator
    from .state import StateManager


def initialize_meta_optimization(
    orchestrator: Orchestrator,
    state_manager: StateManager,
    enable_transfer_learning: bool = True,
    enable_ab_testing: bool = True,
    enable_hitl: bool = True,
    enable_rollout: bool = True,
    storage_path: Path | None = None,
) -> MetaOptimizationV2Wrapper | None:
    """
    Initialize Meta-Optimization V2 for an orchestrator instance.

    Args:
        orchestrator: The orchestrator instance
        state_manager: State manager for accessing project history
        enable_transfer_learning: Enable cross-project transfer learning
        enable_ab_testing: Enable A/B testing for proposals
        enable_hitl: Enable human-in-the-loop approval
        enable_rollout: Enable gradual rollout
        storage_path: Storage path for meta-optimization data

    Returns:
        Meta-optimization wrapper, or None if initialization fails
    """
    try:
        from .meta_orchestrator import ExecutionArchive
        from .meta_v2_integration import MetaOptimizationV2, MetaV2Config

        # Create archive from state manager
        archive = ExecutionArchive(
            archive_path=storage_path or (
                Path.home() / ".orchestrator_cache" / "meta_v2"
            )
        )

        # Configure meta-optimization
        config = MetaV2Config(
            ab_testing_enabled=enable_ab_testing,
            hitl_enabled=enable_hitl,
            rollout_enabled=enable_rollout,
            storage_path=storage_path,
            min_executions_for_optimization=10,  # Lower for faster feedback
        )

        # Create meta-optimization instance
        meta_v2 = MetaOptimizationV2(
            orchestrator=orchestrator,
            archive=archive,
            config=config,
        )

        # Initialize transfer learning if enabled
        if enable_transfer_learning:
            from .transfer_learning import initialize_transfer_engine
            initialize_transfer_engine(archive, storage_path=storage_path)
            logger.info("Transfer learning enabled")

        logger.info(
            f"Meta-Optimization V2 initialized: "
            f"AB={enable_ab_testing}, HITL={enable_hitl}, Rollout={enable_rollout}"
        )

        return MetaOptimizationV2Wrapper(meta_v2)

    except ImportError as e:
        logger.warning(f"Meta-optimization not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize meta-optimization: {e}")
        return None


class MetaOptimizationV2Wrapper:
    """
    Wrapper for MetaOptimizationV2 with engine-specific integration.

    Provides convenient methods for engine integration.
    """

    def __init__(self, meta_v2: MetaOptimizationV2):
        self.meta_v2 = meta_v2
        self._enabled = True
        self._optimization_count = 0

    async def record_project_completion(self, state: Any):
        """
        Record project completion for meta-optimization.

        Args:
            state: ProjectState from orchestrator
        """
        if not self._enabled:
            return

        try:

            # Convert state to trajectory
            trajectory = self._state_to_trajectory(state)

            # Record completion
            await self.meta_v2.record_project_completion(trajectory)

            # Index for transfer learning
            from .transfer_learning import get_transfer_engine
            transfer_engine = get_transfer_engine()
            if transfer_engine:
                await transfer_engine.index_project(trajectory)

        except Exception as e:
            logger.warning(f"Failed to record project completion: {e}")

    async def maybe_optimize(self):
        """
        Run periodic optimization.

        Call this after every N project completions.
        """
        if not self._enabled:
            return []

        try:
            outcomes = await self.meta_v2.maybe_optimize()

            if outcomes:
                self._optimization_count += 1
                logger.info(
                    f"Meta-optimization cycle {self._optimization_count}: "
                    f"{len(outcomes)} proposals"
                )

                for outcome in outcomes:
                    logger.info(
                        f"  {outcome.decision.value}: "
                        f"{outcome.proposal.description[:50]}..."
                    )

            return outcomes

        except Exception as e:
            logger.warning(f"Meta-optimization failed: {e}")
            return []

    def _state_to_trajectory(self, state: Any) -> ProjectTrajectory:
        """Convert ProjectState to ProjectTrajectory."""
        from .meta_orchestrator import ExecutionRecord, ProjectTrajectory

        # Extract task records from results
        task_records = []
        for task_id, result in state.results.items():
            record = ExecutionRecord(
                task_id=task_id,
                task_type=result.task_type.value if hasattr(result.task_type, 'value') else str(result.task_type),
                model_used=result.model_used if hasattr(result, 'model_used') else "unknown",
                success=result.success if hasattr(result, 'success') else True,
                cost_usd=getattr(result, 'cost_usd', 0.0),
                latency_ms=getattr(result, 'latency_ms', 0.0),
                input_tokens=getattr(result, 'input_tokens', 0),
                output_tokens=getattr(result, 'output_tokens', 0),
                score=getattr(result, 'score', 0.9),
                project_id=state.project_id if hasattr(state, 'project_id') else "",
            )
            task_records.append(record)

        trajectory = ProjectTrajectory(
            project_id=state.project_id if hasattr(state, 'project_id') else "unknown",
            project_description=state.project_description if hasattr(state, 'project_description') else "",
            total_cost=sum(r.cost_usd for r in task_records),
            total_time=state.budget.elapsed_seconds if hasattr(state, 'budget') else 0.0,
            success=state.status.value == "success" if hasattr(state, 'status') else True,
            task_records=task_records,
            model_sequence=[r.model_used for r in task_records],
        )

        return trajectory

    def enable(self):
        """Enable meta-optimization."""
        self._enabled = True
        logger.info("Meta-optimization enabled")

    def disable(self):
        """Disable meta-optimization."""
        self._enabled = False
        logger.info("Meta-optimization disabled")

    def get_status(self) -> dict:
        """Get meta-optimization status."""
        status = self.meta_v2.get_status()
        status["enabled"] = self._enabled
        status["optimization_count"] = self._optimization_count
        return status


# ─────────────────────────────────────────────
# Engine Hooks
# ─────────────────────────────────────────────

async def on_task_completed(
    meta_wrapper: MetaOptimizationV2Wrapper | None,
    task_id: str,
    result: Any,
):
    """
    Hook called when a task completes.

    Records task outcome for meta-optimization.
    """
    if not meta_wrapper or not meta_wrapper._enabled:
        return

    # Task-level recording would go here if needed
    # For now, we only record at project completion
    pass


async def on_project_completed(
    meta_wrapper: MetaOptimizationV2Wrapper | None,
    state: Any,
    run_optimization: bool = True,
):
    """
    Hook called when a project completes.

    Records project completion and optionally runs optimization.
    """
    if not meta_wrapper:
        return state

    # Record completion
    await meta_wrapper.record_project_completion(state)

    # Run optimization if requested
    if run_optimization:
        await meta_wrapper.maybe_optimize()

    return state


def get_meta_status(meta_wrapper: MetaOptimizationV2Wrapper | None) -> dict:
    """
    Get meta-optimization status for reporting.

    Returns:
        Status dictionary for dashboard/CLI
    """
    if not meta_wrapper:
        return {"enabled": False, "available": False}

    return meta_wrapper.get_status()
