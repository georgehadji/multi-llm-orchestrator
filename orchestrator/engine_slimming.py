"""
Engine Slimming — Helper factories and delegates for Orchestrator.__init__
===========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Extracted from ``Orchestrator.__init__`` (Phase 2 — God Module Split) to
reduce engine.py from ~1,289 lines to <500 lines.

Holds:
- ``build_health_tracker()`` — factory for ModelHealthTracker
- ``build_project_runner_callables()`` — factory for ProjectRunnerCallables
- ``build_resumption_service()`` — factory for ResumptionService
- ``build_skill_manager()`` — factory for SkillManager
- ``build_taste_skill_service()`` — factory for TasteSkillService
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .api_clients import APIResponse
    from .budget import Budget
    from .cache import DiskCache
    from .circuit_breaker import CircuitBreakerRegistry
    from .engine_core.container import ServiceContainer
    from .engine_core.pipeline_executor import PipelineExecutor
    from .engine_core.project_planner import ProjectPlanner
    from .engine_core.state_coordinator import StateCoordinator
    from .events import EventBus
    from .models import Model, ProjectState, ProjectStatus, Task, TaskResult, TaskType
    from .operations.resilience import ResiliencePolicy
    from .policy import PolicySet
    from .telemetry import TelemetryCollector

logger = logging.getLogger("orchestrator.engine_slimming")


def build_health_tracker(
    container: Any,
    telemetry: Any,
    dashboard_bridge: Any,
    adaptive_router: Any,
    state_mgr: Any,
    circuit_breaker_threshold: int,
    initial_consecutive_failures: dict,
    initial_api_health: dict,
) -> Any:
    """Factory for ModelHealthTracker — wraps circuit breaker + telemetry recording."""
    from .application.model_health_tracker import ModelHealthTracker as _ModelHealthTracker

    return _ModelHealthTracker(
        telemetry=telemetry,
        dashboard=dashboard_bridge,
        adaptive_router=adaptive_router,
        state_mgr=state_mgr,
        circuit_breaker_threshold=circuit_breaker_threshold,
        initial_consecutive_failures=initial_consecutive_failures,
        initial_api_health=initial_api_health,
    )


def build_resumption_service(
    budget: Any,
    results: dict[str, Any],
    execute_task_fn: Any,
    determine_final_status_fn: Any,
) -> Any:
    """Factory for ResumptionService."""
    from .application.resumption_service import ResumptionService as _ResumptionService

    return _ResumptionService(
        budget=budget,
        results=results,
        execute_task_fn=execute_task_fn,
        determine_final_status_fn=determine_final_status_fn,
    )


def build_project_runner_callables(
    topological_sort: Any,
    topological_levels: Any,
    make_state: Any,
    determine_final_status: Any,
    log_summary: Any,
    execute_all: Any,
    generate_architecture_rules: Any,
    analyze_completed_project: Any,
    client: Any,
    warm_start_fn: Any,
    flush_telemetry_fn: Any,
    constitution_gate: Any,
) -> Any:
    """Factory for ProjectRunnerCallables — bundles all callbacks needed by ProjectRunner."""
    from .application.project_runner_deps import ProjectRunnerCallables as _Callables

    return _Callables(
        topological_sort=topological_sort,
        topological_levels=topological_levels,
        make_state=make_state,
        determine_final_status=determine_final_status,
        log_summary=log_summary,
        execute_all=execute_all,
        generate_architecture_rules=generate_architecture_rules,
        analyze_completed_project=analyze_completed_project,
        client=client,
        warm_start_fn=warm_start_fn,
        flush_telemetry_fn=flush_telemetry_fn,
        constitution_gate=constitution_gate,
    )


def build_skill_manager(container: Any, flags: Any) -> Any:
    """Factory for SkillManager — returns None if skill optimization is disabled."""
    if flags.skill_optimization_enabled:
        try:
            from .application.skill_manager import SkillManager as _SkillManager

            skill_store = container.skill_store
            return _SkillManager(
                optimizer_client=container.client,
                skill_store=skill_store,
            )
        except Exception as e:
            logger.warning("Failed to initialize SkillManager: %s", e)
    return None


def build_taste_skill_service(flags: Any, settings: Any) -> Any:
    """Factory for TasteSkillService — anti-slop design prefix for frontend tasks."""
    from .design.taste_skill_service import TasteSkillService as _TasteSkillService

    return _TasteSkillService(flags=flags, settings=settings)


def build_dashboard_bridge(dashboard_integration: Any) -> Any:
    """Factory for DashboardBridge — optional UI integration."""
    from .application.dashboard_bridge import DashboardBridge as _DashboardBridge

    return _DashboardBridge(dashboard_integration)


def build_git_bridge(git_integration: Any) -> Any:
    """Factory for GitBridge — optional VCS integration."""
    from .application.git_bridge import GitBridge as _GitBridge

    return _GitBridge(git_integration)


def build_run_context(budget: Any) -> Any:
    """Factory for RunContext — shared mutable state hub."""
    from .application.run_context import RunContext as _RunCtx

    return _RunCtx(budget=budget)


def build_meta_v2(container: Any, state_manager: Any) -> Any:
    """Factory for meta_v2 — transfer learning, A/B testing, HITL, rollout."""
    from .meta_integration import initialize_meta_optimization

    # Pass a minimal proxy to avoid circular init — engine is not yet fully constructed
    class _EngineProxy:
        def __init__(self, container_obj: Any):
            self._c = container_obj
            self._telemetry = container_obj.telemetry

    return initialize_meta_optimization(
        orchestrator=_EngineProxy(container),
        state_manager=state_manager,
        enable_transfer_learning=True,
        enable_ab_testing=True,
        enable_hitl=True,
        enable_rollout=True,
    )
