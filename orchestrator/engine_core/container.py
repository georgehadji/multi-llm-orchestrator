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
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .api_clients import UnifiedClient
from .budget import Budget
from .cost_optimization_integration import (
    AdaptiveTemperatureController,
    BatchClient,
    DependencyContextInjector,
    PromptCacher,
    SpeculativeGenerator,
    StreamingValidator,
    TokenBudget,
)
from .model_registry import ModelRegistry, ModelCascader
from .model_selector import ModelSelector
from .policy_engine import PolicyEngine
from .rate_limiter import RateLimiter
from .telemetry import TelemetryCollector
from .tracing import Tracer

if TYPE_CHECKING:
    pass

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
    cache: Any
    state_mgr: Any

    # Core services
    task_guard: Any
    results_lock: asyncio.Lock
    selector: ModelSelector
    telemetry: TelemetryCollector
    tracer: Tracer
    policy_engine: PolicyEngine
    planner: Any
    preflight_validator: Any
    hook_registry: Any
    validator: Any
    decomposer: Any
    architect: Any
    executor: Any
    evaluator: Any
    generator: Any
    ara: Any
    ara_strategy: Any
    pipeline: Any
    dep_resolver: Any
    event_bus: Any
    adaptive_router: Any
    telemetry_store: Any
    semantic_cache: Any
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
        from .audit import AuditLog
        from .engine_deps import (
            _CBRegistry as CBRegistry,
            _DepResolver as DepResolver,
            _ExecutorService as ExecutorService,
            _GeneratorService as GeneratorService,
        )
        from .engine_core.architect import Architect
        from .engine_core.decomposer import Decomposer
        from .engine_core.pipeline import TaskPipeline
        from .engine_core.stages import (
            CritiqueStage,
            EvaluateStage,
            GenerateStage,
            PersuasionDefenseStage,
            PreflightStage,
            SelfConsistencyStage,
            ValidateStage,
        )
        from .engine_core.validator import TaskValidator
        from .evaluator_service import EvaluatorService
        from .models import Model, TaskType
        from .output_organizer import OutputOrganizer
        from .planner import ConstraintPlanner
        from .preflight import PreflightValidator
        from .state import StateManager, TelemetryStore
        from .task_guard import TaskGuard

        # Defaults
        if cache is None:
            from .cache import DiskCache
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

        profiles: dict[Model, Any] = {}
        try:
            from .models import ModelProfile
            for m in Model:
                profiles[m] = ModelProfile(model=m)
        except ImportError:
            pass

        telemetry = TelemetryCollector(profiles)
        api_health: dict[Model, bool] = {}

        # Wired helpers
        planner = ConstraintPlanner(profiles=profiles, policy_engine=policy_engine)

        def get_available_models(task_type: object = None) -> list[Model]:
            from .models import ROUTING_TABLE
            routing = ROUTING_TABLE.get(task_type, [])
            return [m for m in routing if api_health.get(m, True)]

        selector = ModelSelector(api_health, get_available_models)

        # Services (wrapped in type: ignore for optional dependencies)
        executor = ExecutorService(
            execute_fn=None,  # assigned later by Orchestrator
            guard=task_guard,
            telemetry=telemetry,
        )
        evaluator = EvaluatorService(
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
        hook_registry = type('HookRegistry', (), {'fire': lambda *a, **kw: None})()

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
        pipeline = TaskPipeline([
            GenerateStage(client=client, budget=budget, selector=selector),
            CritiqueStage(client=client),
            EvaluateStage(evaluator=evaluator),
            ValidateStage(),
            PersuasionDefenseStage(ara_integration=ara),
            PreflightStage(validator=validator),
            SelfConsistencyStage(
                max_attempts=2,
                quality_threshold=0.7,
                ara_strategy=ara_strategy,
            ),
        ])

        # State management
        if telemetry_store is None:
            telemetry_store = TelemetryStore()

        # Dependency resolver
        dep_resolver = DepResolver(context_truncation_limit=8192)

        # Event bus
        event_bus = type('EventBus', (), {'publish': lambda *a, **kw: None})()

        # Semantic cache
        semantic_cache = type('SemanticCache', (), {
            'get_cached_pattern': lambda *a: None,
            'cache_pattern': lambda *a: None,
        })()

        # Adaptive router
        adaptive_router = type('AdaptiveRouter', (), {})()

        return cls(
            budget=budget,
            client=client,
            cache=cache,
            state_mgr=state_manager,
            task_guard=task_guard,
            results_lock=results_lock,
            selector=selector,
            telemetry=telemetry,
            tracer=tracer,
            policy_engine=policy_engine,
            planner=planner,
            preflight_validator=preflight_validator,
            hook_registry=hook_registry,
            validator=validator,
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
            telemetry_store=telemetry_store,
            semantic_cache=semantic_cache,
            cache_optimizer=None,
            cost_predictor=cost_predictor,
            budget_hierarchy=budget_hierarchy,
            output_dir=output_dir,
            git_integration=None,
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
