"""
ServiceContainer — Factory for all Orchestrator collaborators
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 5 of the Master Architecture Enhancement Plan.
Centralizes the wiring of 30+ collaborators that were previously
inline in Orchestrator.__init__.

Usage:
    container = ServiceContainer.build(budget=budget, cache=cache, ...)
    orch = Orchestrator(container=container)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .pipeline import TaskPipeline
    from .pipeline_runner import PipelineRunner
    from .project_planner import ProjectPlanner
    from .state_coordinator import StateCoordinator
    from .context_service import ContextService

from ..api_clients import UnifiedClient
from ..budget import Budget
from ..domain.ports import (
    CachePort,
    EventPort,
    HookRegistryPort,
    NullEventBus,
    NullHookRegistry,
    PlannerPort,
    StatePort,
    ValidatorPort,
)

try:
    from ..cost_optimization import (  # type: ignore[attr-defined]
        AdaptiveTemperatureController,
        BatchClient,
        DependencyContextInjector,
        PromptCacher,
        SpeculativeGenerator,
        StreamingValidator,
        TokenBudget,
    )
except ImportError:
    AdaptiveTemperatureController = None
    BatchClient = None
    DependencyContextInjector = None
    PromptCacher = None
    SpeculativeGenerator = None
    StreamingValidator = None
    TokenBudget = None
from ..model_registry import ModelRegistry

try:
    from ..model_registry import ModelCascader  # type: ignore[attr-defined]
except ImportError:
    ModelCascader = None
from ..model_selector import ModelSelector, TieredModelRouter
from ..policy_engine import PolicyEngine
from ..rate_limiter import RateLimiter
from ..telemetry import TelemetryCollector
from ..tracing import Tracer

logger = logging.getLogger("orchestrator.container")


@dataclass
class ServiceContainer:
    """Holds all wired collaborators for the Orchestrator.

    Every service/component that the Orchestrator needs to function
    is wired here via the factory method ``build()``.

    Lazy-init attributes (accessed via ``@property``) are not included
    here — they remain property-based for efficiency.
    """

    budget: Budget
    client: UnifiedClient
    cache: CachePort
    state_mgr: StatePort

    # Core services
    task_guard: Any
    results_lock: asyncio.Lock
    selector: ModelSelector  # also satisfies PlannerPort
    tiered_router: TieredModelRouter = None
    telemetry: TelemetryCollector = None
    tracer: Tracer = None
    policy_engine: PolicyEngine = None
    planner: Any = None
    project_planner: Optional[ProjectPlanner] = None
    pipeline_runner: Optional[PipelineRunner] = None
    preflight_validator: Any = None
    hook_registry: Optional[HookRegistryPort] = None
    validator: Optional[ValidatorPort] = None
    decomposer: Any = None
    architect: Any = None
    executor: Any = None
    evaluator: Any = None
    generator: Any = None
    ara: Any = None
    ara_strategy: Any = None
    pipeline: Optional[TaskPipeline] = None
    dep_resolver: Any = None
    event_bus: Optional[EventPort] = None
    adaptive_router: Any = None
    telemetry_store: Any = None
    semantic_cache: Any = None
    cb_registry: Any = None
    observability: Any = None

    # Extracted from engine.__init__ to centralize wiring
    skill_manager: Any = None
    skill_store: Any = None
    taste_skill_service: Any = None
    dashboard_bridge: Any = None
    git_bridge: Any = None
    context_compressor: Any = None
    memory_provider_mgr: Any = None
    pattern_store: Any = None
    pattern_extractor: Any = None
    pattern_injector: Any = None
    pattern_curator: Any = None
    batch_guard: Any = None
    batch_runner: Any = None
    tool_guardrails: Any = None
    optim_config: Any = None
    meta_v2: Any = None

    # New Domain Services (Phase 2 refactor)
    routing_service: Any = None
    cost_service: Any = None
    config_service: Any = None
    state_coordinator: Optional[StateCoordinator] = None
    context_service: Optional[ContextService] = None

    git_integration: Any = None
    output_dir: Path | None = None

    # Dashboard
    dashboard_integration: Any = None

    # Cost optimization
    cache_optimizer: Any = None
    cost_predictor: Any = None
    budget_hierarchy: Any = None
    prompt_cacher: Any = None
    batch_client: Any = None
    token_budget: Any = None
    model_cascader: Any = None
    speculative_gen: Any = None
    streaming_validator: Any = None
    dependency_injector: Any = None
    adaptive_temp: Any = None
    eval_dataset: Any = None
    tdd_generator: Any = None
    diff_generator: Any = None

    # Accessory services (rarely used, kept here for reference)
    session_watcher: Any = None
    persona_manager: Any = None
    memory_manager: Any = None
    bm25_search: Any = None
    reranker: Any = None
    hybrid_pipeline: Any = None
    a2a_manager: Any = None
    rate_limiter: Any = None
    lifecycle_manager: Any = None
    knowledge_base: Any = None
    query_expander: Any = None
    token_optimizer: Any = None
    red_team: Any = None
    task_verifier: Any = None
    accountability: Any = None
    agent_safety: Any = None

    async def shutdown(self) -> None:
        """Release all container-managed resources.

        Order matters — dependent services are shut down before their providers.
        After this call, the container should not be reused.
        """
        logger.debug("ServiceContainer shutting down...")

        # 1. Stop session lifecycle scheduler (no-op if never started)
        if self.lifecycle_manager is not None:
            try:
                if hasattr(self.lifecycle_manager, "stop"):
                    await self.lifecycle_manager.stop()
            except Exception as e:
                logger.warning("Failed to stop lifecycle manager: %s", e)

        # 2. Flush telemetry store
        if self.telemetry_store is not None:
            try:
                if hasattr(self.telemetry_store, "flush"):
                    await self.telemetry_store.flush()
            except Exception as e:
                logger.warning("Failed to flush telemetry store: %s", e)

        # 3. Close cache (aiosqlite background thread yield)
        if self.cache is not None:
            try:
                await self.cache.close()
                await asyncio.sleep(0)
            except Exception as e:
                logger.warning("Failed to close cache: %s", e)

        # 4. Close state manager (aiosqlite background thread yield)
        if self.state_mgr is not None:
            try:
                await self.state_mgr.close()
                await asyncio.sleep(0)
            except Exception as e:
                logger.warning("Failed to close state manager: %s", e)

        # 5. Close semantic cache
        if self.semantic_cache is not None:
            try:
                if hasattr(self.semantic_cache, "close"):
                    await self.semantic_cache.close()
            except Exception as e:
                logger.warning("Failed to close semantic cache: %s", e)

        # 6. Close event bus
        if self.event_bus is not None:
            try:
                if hasattr(self.event_bus, "close"):
                    await self.event_bus.close()
            except Exception as e:
                logger.warning("Failed to close event bus: %s", e)

    # Lazy-init cache for optional services. Managed by get_or_create().
    _lazy_cache: dict[str, Any] = field(default_factory=dict)

    def get_or_create(self, name: str, factory: Callable[[], Any]) -> Any:
        """Lazy-init cache for optional services.

        Returns a cached instance if already created, otherwise calls factory,
        stores the result, and returns it. Safe under asyncio's single-threaded
        event loop (synchronous factory is atomic between check and set).
        """
        if name not in self._lazy_cache:
            self._lazy_cache[name] = factory()
        return self._lazy_cache[name]

    def wire_executor(self, execute_fn: Any, decompose_fn: Any = None) -> None:
        """Late-bind execute_fn and decompose_fn after Orchestrator.__init__ creates them."""
        if self.executor is not None and hasattr(self.executor, "execute_fn"):
            self.executor.execute_fn = execute_fn
        if (
            self.generator is not None
            and decompose_fn is not None
            and hasattr(self.generator, "decompose_fn")
        ):
            self.generator.decompose_fn = decompose_fn

    @classmethod
    def build(
        cls,
        budget: Budget,
        cache: Any | None = None,
        state_manager: Any | None = None,
        max_concurrency: int = 3,
        max_parallel_tasks: int = 3,
        output_dir: Path | None = None,
        telemetry_store: Any | None = None,
        budget_hierarchy: Any | None = None,
        cost_predictor: Any | None = None,
        project_context: Any | None = None,
        profiles: Any | None = None,
    ) -> ServiceContainer:
        """Factory: wire all services with their dependencies.

        This is the single place where the dependency graph is assembled.
        Callers can override individual services for testing or customization.

        Args:
            budget: Budget tracker for cost enforcement.
            cache: Optional cache adapter (defaults to DiskCache).
            state_manager: Optional state adapter (defaults to StateManager).
            max_concurrency: Maximum concurrent operations.
            max_parallel_tasks: Maximum parallel tasks.
            output_dir: Directory for output files.
            telemetry_store: Optional telemetry persistence.
            budget_hierarchy: Optional hierarchical budget.
            cost_predictor: Optional cost prediction.
            project_context: Optional project context accumulator.

        Returns:
            Fully wired ServiceContainer.
        """
        from ..audit import AuditLog
        from ..application.executor import ExecutorService
        from ..application.evaluator import EvaluatorService
        from .decomposer import Decomposer
        from .architect import Architect
        from .pipeline import TaskPipeline
        from .stages import (
            CritiqueStage,
            EvaluateStage,
            GenerateStage,
            PersuasionDefenseStage,
            PreflightStage,
            SelfConsistencyStage,
            ValidateStage,
        )
        from .validator import TaskValidator
        from ..models import Model, TaskType
        from ..output_organizer import OutputOrganizer
        from ..planner import ConstraintPlanner
        from ..preflight import PreflightValidator
        from ..state import StateManager
        from ..telemetry_store import TelemetryStore

        try:
            from ..task_guard import TaskGuard
        except ImportError:
            from ..concurrency_controller import TaskConcurrencyGuard as TaskGuard

        # Import canonical GeneratorService from services layer
        try:
            from ..services import GeneratorService
        except ImportError:
            GeneratorService = None  # type: ignore[misc]

        # Local shims for legacy deps if not found in application layer
        try:
            from .engine_deps import (  # type: ignore[attr-defined]
                _CBRegistry as CBRegistry,
                _DepResolver as DepResolver,
                # GeneratorService imported from services layer above
            )
        except ImportError:
            # Fallback for missing deps
            class _DepResolver:
                def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                    pass

            DepResolver = _DepResolver
            # CBRegistry is wired separately via try/except below; not needed here

        # Defaults
        if cache is None:
            from ..infrastructure.cache import DiskCache

            cache = DiskCache()
        if state_manager is None:
            state_manager = StateManager()

        # Core infrastructure
        client = UnifiedClient(cache=cache)
        results_lock = asyncio.Lock()
        task_guard = TaskGuard(name="tasks", max_concurrent=max_concurrency)
        tracer = Tracer("orchestrator")
        audit_log = AuditLog()
        policy_engine = PolicyEngine(audit_log=audit_log)
        models_dict = dict.fromkeys(Model, None)

        profiles: dict[Model, Any] = {}  # type: ignore[no-redef]
        try:
            from .models import ModelProfile

            for m in Model:
                profiles[m] = ModelProfile(model=m)
        except ImportError:
            pass

        telemetry = TelemetryCollector(profiles)
        api_health: dict[Model, bool] = {}

        # New Domain Services (Phase 2 refactor)
        try:
            from ..infrastructure.adapters.config_adapter import JsonConfigAdapter
            from ..domain.services.config_services import (
                RoutingService,
                CostService,
                ConfigurationService,
            )

            config_adapter = JsonConfigAdapter()
            routing_service = RoutingService(config_adapter)
            cost_service = CostService(config_adapter)
            config_service = ConfigurationService(config_adapter)

            from .state_coordinator import StateCoordinator
            from .context_service import ContextService

            state_coordinator = StateCoordinator()
            context_service = ContextService()
        except ImportError:
            logger.warning("Failed to load new domain services (Phase 2)")
            routing_service = None
            cost_service = None
            config_service = None
            state_coordinator = None
            context_service = None

        # Wired helpers
        planner = ConstraintPlanner(
            profiles=profiles, policy_engine=policy_engine, api_health=api_health
        )

        selector = ModelSelector(
            api_health=api_health,
            routing_service=routing_service,
            cost_service=cost_service,
        )

        tiered_router = TieredModelRouter(
            api_health=api_health,
            routing_service=routing_service,
            cost_service=cost_service,
            adaptive_router=None,  # wired later if needed
        )

        def get_available_models(task_type: object = None) -> list[Model]:
            return selector.available_models(task_type)  # type: ignore[arg-type]

        # Services (wrapped in type: ignore for optional dependencies)
        executor = ExecutorService(
            execute_fn=None,  # assigned later by Orchestrator
            guard=task_guard,
            telemetry=telemetry,
        )
        evaluator = EvaluatorService(
            client=client,  # type: ignore[arg-type]
            budget=budget,
            get_models_fn=get_available_models,
            telemetry=telemetry,
        )
        generator = GeneratorService(
            decompose_fn=None,  # assigned later
        )

        # Engine core components
        decomposer = Decomposer(client=client, selector=selector, tracer=tracer)
        architect = Architect(client=client)

        # Preflight + validator
        preflight_validator = PreflightValidator()

        # LSP validator (CodeWhale Phase 1 — post-generation diagnostics)
        lsp_validator = None
        try:
            from ..infrastructure.lsp_validator import LspValidator

            lsp_validator = LspValidator(timeout_seconds=30)
            known = lsp_validator.available_servers()
            if known:
                logger.info("LSP servers available: %s", ", ".join(sorted(known)))
            else:
                logger.info("No LSP servers found on PATH — LSP validation disabled")
                lsp_validator = None
        except Exception as exc:
            logger.debug("LspValidator not available: %s", exc)

        # Unified Event System integration
        try:
            from .unified_events.core import UnifiedEventBus

            # event_bus  → async face (publish DomainEvents)
            # hook_registry → sync face (fire lifecycle hooks)
            # They share the same underlying object; .sync_hooks just narrows the type.
            event_bus = UnifiedEventBus()
            hook_registry = event_bus.sync_hooks
        except ImportError:
            # Fallback to domain null-adapters if unified_events is not reachable
            hook_registry = NullHookRegistry()
            event_bus = NullEventBus()

        validator = TaskValidator(
            client=client,
            budget=budget,
            preflight_validator=preflight_validator,
            hook_registry=hook_registry,
        )

        # ARA integration
        try:
            from .ara_integration import create_ara_integration

            ara = create_ara_integration(
                client=client,
                cache=cache,
                telemetry=telemetry,
                enabled=True,
                auto_select=True,
            )
        except ImportError:
            ara = None

        try:
            from .ara_execution_strategy import ARAExecutionStrategy

            ara_strategy = ARAExecutionStrategy(ara_integration=ara)
        except ImportError:
            ara_strategy = None

        # Pipeline with all stages
        pipeline = TaskPipeline(
            [
                GenerateStage(client=client, budget=budget, selector=selector),  # type: ignore[arg-type]
                CritiqueStage(client=client, lsp_validator=lsp_validator),  # type: ignore[arg-type]
                EvaluateStage(evaluator=evaluator),
                ValidateStage(),
                PersuasionDefenseStage(ara_integration=ara),
                PreflightStage(validator=validator),
                SelfConsistencyStage(
                    max_attempts=2,
                    quality_threshold=0.7,
                    ara_strategy=ara_strategy,
                ),
            ]
        )

        # State management
        if telemetry_store is None:
            telemetry_store = TelemetryStore()

        # Dependency resolver
        dep_resolver = DepResolver(context_truncation_limit=8192)

        # New Domain Services (Phase 3 refactor: Planning and Execution)
        from .project_planner import ProjectPlanner
        from .pipeline_runner import PipelineRunner

        project_planner = ProjectPlanner(dep_resolver=dep_resolver)
        pipeline_runner = PipelineRunner(
            pipeline=pipeline,
            planner=project_planner,
            max_parallel_tasks=max_parallel_tasks,
            event_bus=event_bus,
            dashboard=None,  # wired later
        )

        # Semantic cache — real SemanticCache wired by Orchestrator if needed;
        # None here so callers must guard with `if self._semantic_cache is not None`.
        semantic_cache = None

        # Adaptive router — real AdaptiveRouter wired by Orchestrator if needed;
        # None here so callers must guard with `if self._adaptive_router is not None`.
        adaptive_router = None

        # NexusScope: wrap pipeline with profiling if enabled
        import os as _os

        if _os.getenv("ORCHESTRATOR_PROFILING", "0") == "1":
            try:
                from ..infrastructure.nexusscope.pipeline_hook import ProfilingTaskPipeline as _PTP
                from ..infrastructure.nexusscope import get_profiler as _get_ns

                pipeline = _PTP(pipeline._stages, profiler=_get_ns())  # type: ignore[assignment]
            except ImportError:
                pass

        # Observability service
        observability = None
        try:
            from ..services import ObservabilityService

            observability = ObservabilityService()
        except ImportError:
            pass

        # Circuit breaker registry
        cb_registry = None
        try:
            from ..circuit_breaker import CircuitBreakerRegistry

            cb_registry = CircuitBreakerRegistry()
        except ImportError:
            pass

        # ── Services extracted from engine.__init__ (Phase C.3) ────────
        from ..crosscutting.config import flags as _flags
        from ..crosscutting.config import settings as _settings

        # SkillOpt: self-improving per-TaskType skill documents
        skill_manager = None
        skill_store = None
        if _flags.skill_optimization_enabled:
            from ..application.skill_store import SkillStore as _SkillStore
            from ..application.skill_manager import SkillManager as _SkillManager

            skill_store = _SkillStore()
            skill_manager = _SkillManager(
                optimizer_client=client,
                skill_store=skill_store,
            )

        # Taste-skill: anti-slop design prefix for frontend tasks
        taste_skill_service = None
        try:
            from ..design.taste_skill_service import TasteSkillService as _TasteSkillService

            taste_skill_service = _TasteSkillService(flags=_flags, settings=_settings)
        except ImportError:
            pass

        # Thin bridge wrappers (dashboard/git integrations default to None)
        from ..application.dashboard_bridge import DashboardBridge as _DashboardBridge
        from ..application.git_bridge import GitBridge as _GitBridge

        dashboard_bridge = _DashboardBridge(None)
        git_bridge = _GitBridge(None)

        return cls(
            budget=budget,
            client=client,
            cache=cache,
            state_mgr=state_manager,
            task_guard=task_guard,
            results_lock=results_lock,
            selector=selector,
            tiered_router=tiered_router,
            telemetry=telemetry,
            tracer=tracer,
            policy_engine=policy_engine,
            planner=planner,
            preflight_validator=preflight_validator,
            hook_registry=hook_registry,
            validator=validator,  # type: ignore[arg-type]
            decomposer=decomposer,
            architect=architect,
            executor=executor,
            evaluator=evaluator,
            generator=generator,
            ara=ara,
            ara_strategy=ara_strategy,
            pipeline=pipeline,
            dep_resolver=dep_resolver,
            event_bus=event_bus,
            adaptive_router=adaptive_router,
            routing_service=routing_service,
            cost_service=cost_service,
            config_service=config_service,
            state_coordinator=state_coordinator,
            context_service=context_service,
            project_planner=project_planner,
            pipeline_runner=pipeline_runner,
            telemetry_store=telemetry_store,
            semantic_cache=semantic_cache,
            cache_optimizer=None,
            cost_predictor=cost_predictor,
            budget_hierarchy=budget_hierarchy,
            output_dir=output_dir,
            git_integration=None,
            observability=observability,
            cb_registry=cb_registry,
            skill_manager=skill_manager,
            skill_store=skill_store,
            taste_skill_service=taste_skill_service,
            dashboard_bridge=dashboard_bridge,
            git_bridge=git_bridge,
        )


# Lazy-init helpers for accessory services
class LazyServices:
    """Container for rarely-used accessory services.

    These are initialized on first access (property pattern) rather than
    eagerly during Orchestrator.__init__.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, name: str, factory: type) -> Any:
        if name not in self._cache:
            self._cache[name] = factory()
        return self._cache[name]
