"""
Orchestrator Engine — Core Control Loop
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Implements the full generate → critique → revise → evaluate pipeline
with cross-model review, deterministic validation, budget enforcement,
plateau detection, and fallback routing.

FIX #5:  Budget checked within iteration loop (mid-task), not just pre-task.
FIX #6:  Topological sort uses collections.deque instead of list.sort()+pop(0).
FIX #7:  Resume restores persisted budget state instead of creating fresh Budget.
FIX #10: All StateManager calls are now awaited (async migration).
FEAT:    TelemetryCollector + ConstraintPlanner wired at init.
FEAT:    TaskResult.tokens_used populated from APIResponse.
FEAT:    run_job(spec) entry point for policy-driven orchestration.
FEAT:    Budget phase partition enforcement (warn + soft-halt at 2× soft cap).
FEAT:    Dependency context truncation warning.
FEAT:    Decomposition retried once with different model on JSON parse failure.
FEAT:    Circuit breaker — model marked unhealthy after 3 consecutive failures.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .api_clients import APIResponse
from .prompt_builder import (
    RevisionPrompt,
)
from .operations.autonomy_config import AutonomyConfig, AutonomyLevel
from .budget import Budget
from .cache import DiskCache
from .models import (
    Model,
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskType,
    get_provider,
)

from .resilience import ResiliencePolicy
from .exceptions import (
    ConfigurationError,
)
from .application.validators import validate_job_spec, validate_project_args
from .crosscutting.config import flags
from .engine_flags import import_feature_modules
from .engine_slimming import (
    build_health_tracker,
    build_resumption_service,
    build_project_runner_callables,
    build_skill_manager,
    build_taste_skill_service,
    build_dashboard_bridge,
    build_git_bridge,
    build_run_context,
    build_meta_v2,
)

# OpenRouter Optimization Features (Phase 1)
try:
    from .config import OPENROUTER_OPTS
except ImportError:
    OPENROUTER_OPTS = None
try:
    from .task_schemas import generate_openrouter_schema, get_schema_for_task_type
except ImportError:
    generate_openrouter_schema = None
    get_schema_for_task_type = None

# P4-2: Optional imports are gated by FeatureFlags so the feature surface is

_feat = import_feature_modules(flags)

# explicit and testable.  Each block checks its flag first; if the flag is
# disabled the import is never attempted and the symbols are set to None.
# All flags default to True to preserve existing behaviour; users can opt out
# via environment variables (e.g. ORCH_A2A_ENABLED=false).

CacheOptimizer = _feat.get("CacheOptimizer")
CacheConfig = _feat.get("CacheConfig")
HAS_CACHE_OPTIMIZER = CacheOptimizer is not None

from .policy import JobSpec, Policy, PolicySet
from .state import StateManager

# Test validation for reliable test generation
if flags.test_validation_enabled:
    try:
        from .test_validator import TestValidator, validate_and_generate_test

        HAS_TEST_VALIDATOR = True
    except ImportError:
        HAS_TEST_VALIDATOR = False
        TestValidator = None
        validate_and_generate_test = None
else:
    HAS_TEST_VALIDATOR = False
    TestValidator = None
    validate_and_generate_test = None

# Code validation for clean code generation (no LLM commentary)
if flags.code_validation_enabled:
    try:
        from .code_validator import validate_code, extract_code_from_llm_response

        HAS_CODE_VALIDATOR = True
    except ImportError:
        HAS_CODE_VALIDATOR = False
        validate_code = None
        extract_code_from_llm_response = None
else:
    HAS_CODE_VALIDATOR = False
    validate_code = None
    extract_code_from_llm_response = None

# Optional advanced features — gated by feature flags (P4-2)
HAS_ADVANCED_FEATURES = True

if flags.a2a_enabled:
    try:
        from .a2a_protocol import A2AManager, AgentCard
    except (ImportError, TimeoutError):
        A2AManager = None
        AgentCard = None
        HAS_ADVANCED_FEATURES = False
else:
    A2AManager = None
    AgentCard = None

if flags.accountability_enabled:
    try:
        from .accountability import AccountabilityTracker, ActionType, ActorType
    except (ImportError, TimeoutError):
        AccountabilityTracker = None
        ActionType = None
        ActorType = None
else:
    AccountabilityTracker = None
    ActionType = None
    ActorType = None

if flags.agent_safety_enabled:
    try:
        from .agent_safety import AgentSafetyMonitor, SafetyEventType
    except (ImportError, TimeoutError):
        AgentSafetyMonitor = None
        SafetyEventType = None
else:
    AgentSafetyMonitor = None
    SafetyEventType = None

try:
    from .agents import TaskChannel
except (ImportError, TimeoutError):
    TaskChannel = None

if flags.audit_log:
    try:
        from .audit import AuditLog
    except (ImportError, TimeoutError):
        AuditLog = None
else:
    AuditLog = None

if flags.bm25_search_enabled:
    try:
        from .bm25_search import BM25Search, get_bm25_search
    except (ImportError, TimeoutError):
        BM25Search = None
        get_bm25_search = None
else:
    BM25Search = None
    get_bm25_search = None

# OPTIMIZATION: Cost & performance optimizations (Tiers 1-4)
if flags.cost_optimization_enabled:
    try:
        from .cost_optimization import (
            AdaptiveTemperatureController,
            BatchClient,
            DependencyContextInjector,
            EvalDatasetBuilder,
            ModelCascader,
            OptimizationConfig,
            PromptCacher,
            SpeculativeGenerator,
            StreamingValidator,
            TokenBudget,
            cascading_generate,
            get_optimization_config,
            inject_dependency_context,
            speculative_generate,
            stream_and_validate,
        )
    except (ImportError, TimeoutError):
        OptimizationConfig = None
        get_optimization_config = None
        AdaptiveTemperatureController = None
        BatchClient = None
        DependencyContextInjector = None
        EvalDatasetBuilder = None
        ModelCascader = None
        PromptCacher = None
        SpeculativeGenerator = None
        StreamingValidator = None
        TokenBudget = None
        cascading_generate = None
        inject_dependency_context = None
        speculative_generate = None
        stream_and_validate = None
else:
    OptimizationConfig = None
    get_optimization_config = None
    AdaptiveTemperatureController = None
    BatchClient = None
    DependencyContextInjector = None
    EvalDatasetBuilder = None
    ModelCascader = None
    PromptCacher = None
    SpeculativeGenerator = None
    StreamingValidator = None
    TokenBudget = None
    cascading_generate = None
    inject_dependency_context = None
    speculative_generate = None
    stream_and_validate = None

try:
    from .hooks import EventType, HookRegistry
except (ImportError, TimeoutError):
    EventType = None
    HookRegistry = None

if flags.memory_tier_enabled:
    try:
        from .memory_tier import MemoryTierManager
    except (ImportError, TimeoutError):
        MemoryTierManager = None
else:
    MemoryTierManager = None

if flags.persona_enabled:
    try:
        from .persona import PersonaManager, PersonaMode
    except (ImportError, TimeoutError):
        PersonaManager = None
        PersonaMode = None
else:
    PersonaManager = None
    PersonaMode = None

try:
    from .planner import ConstraintPlanner
except (ImportError, TimeoutError):
    ConstraintPlanner = None

try:
    from .preflight import PreflightMode, PreflightValidator
except (ImportError, TimeoutError):
    PreflightMode = None
    PreflightValidator = None

try:
    from .rate_limiter import RateLimiter
except (ImportError, TimeoutError):
    RateLimiter = None

if flags.red_team_enabled:
    try:
        from .red_team import RedTeamFramework
    except (ImportError, TimeoutError):
        RedTeamFramework = None
else:
    RedTeamFramework = None

if flags.reranker_enabled:
    try:
        from .reranker import LLMReranker, get_reranker
    except (ImportError, TimeoutError):
        LLMReranker = None
        get_reranker = None
else:
    LLMReranker = None
    get_reranker = None

if flags.session_lifecycle_enabled:
    try:
        from .session_lifecycle import SessionLifecycleManager
    except (ImportError, TimeoutError):
        SessionLifecycleManager = None
else:
    SessionLifecycleManager = None

if flags.session_watcher_enabled:
    try:
        from .session_watcher import SessionWatcher
    except (ImportError, TimeoutError):
        SessionWatcher = None
else:
    SessionWatcher = None

# Security & Accountability modules from "Agents of Chaos" paper (arXiv:2602.20021)
if flags.task_verifier_enabled:
    try:
        from .task_verifier import TaskVerifier
    except (ImportError, TimeoutError):
        TaskVerifier = None
else:
    TaskVerifier = None

try:
    from .telemetry import TelemetryCollector
except (ImportError, TimeoutError):
    TelemetryCollector = None

try:
    from .telemetry_store import TelemetryStore
    from .memory.memory_manager import MemoryManager
    from .pattern_learner.pattern_store import PatternStore
    from .pattern_learner.injector import PatternInjector
    from .pattern_learner.extractor import PatternExtractor
    from .pattern_learner.curator import PatternCurator
    from .context_compressor import ContextCompressor
    from .delegation.batch_runner import BatchRunner
except (ImportError, TimeoutError):
    TelemetryStore = None
    MemoryManager = None
    PatternStore = None
    PatternInjector = None
    PatternExtractor = None
    PatternCurator = None
    ContextCompressor = None
    BatchRunner = None

if flags.token_optimizer_enabled:
    try:
        from .token_optimizer import TokenOptimizer
    except (ImportError, TimeoutError):
        TokenOptimizer = None
else:
    TokenOptimizer = None

# OpenTelemetry tracing — disabled by default (requires extra deps)
if flags.tracing_enabled:
    try:
        from .tracing import TracingConfig, configure_tracing, get_tracer, traced_task
    except (ImportError, TimeoutError):
        TracingConfig = None
        configure_tracing = None
        get_tracer = None
        traced_task = None
else:
    TracingConfig = None
    configure_tracing = None
    get_tracer = None
    traced_task = None

if TYPE_CHECKING:
    from .cost import BudgetHierarchy, CostPredictor
    from .metrics import MetricsExporter

# TDD-First and Diff-Based Generation
if flags.tdd_enabled:
    try:
        from .test_first_generator import TDDResult, TestFirstGenerator

        HAS_TDD = True
    except ImportError:
        HAS_TDD = False
        TestFirstGenerator = None
        TDDResult = None
else:
    HAS_TDD = False
    TestFirstGenerator = None
    TDDResult = None

if flags.diff_generation_enabled:
    try:
        from .diff_generator import DiffGenerator, DiffResult, apply_unified_diff

        HAS_DIFF = True
    except ImportError:
        HAS_DIFF = False
        DiffGenerator = None
        DiffResult = None
        apply_unified_diff = None
else:
    HAS_DIFF = False
    DiffGenerator = None
    DiffResult = None
    apply_unified_diff = None

logger = logging.getLogger("orchestrator")


# P3-7: _clean_code_output extracted to orchestrator/output/code_cleaner.py
# This re-export keeps callers inside engine.py unchanged.


class Orchestrator:
    """
    Main orchestration engine.

    Invariants maintained:
    1. Cross-review always uses different provider than generator
    2. Deterministic validators override LLM scores
    3. Budget ceiling is never exceeded (checked mid-task per iteration)
    4. State is checkpointed after each task
    5. Plateau detection prevents runaway iteration
    """

    # Circuit breaker: model is marked unhealthy after this many consecutive errors
    _CIRCUIT_BREAKER_THRESHOLD: int = 3

    def __init__(
        self,
        budget: Budget | None = None,
        cache: "CachePort | DiskCache | None" = None,  # noqa: F821
        state_manager: "StatePort | StateManager | None" = None,  # noqa: F821
        max_concurrency: int = 3,
        max_parallel_tasks: int = 3,
        budget_hierarchy: BudgetHierarchy | None = None,
        cost_predictor: CostPredictor | None = None,
        tracing_cfg: TracingConfig | None = None,
        telemetry_store: TelemetryStore | None = None,
        profiles: dict | None = None,
        container: "ServiceContainer | None" = None,  # noqa: F821
    ):
        from .engine_core.container import ServiceContainer

        if container is None:
            container = ServiceContainer.build(
                budget=budget or Budget(),
                cache=cache,
                state_manager=state_manager,
                max_concurrency=max_concurrency,
                max_parallel_tasks=max_parallel_tasks,
                budget_hierarchy=budget_hierarchy,
                cost_predictor=cost_predictor,
                telemetry_store=telemetry_store,
                profiles=profiles,
            )

        self._c = container
        self.budget = container.budget
        self.cache = container.cache
        self.state_mgr = container.state_mgr
        self.client = container.client
        self._task_guard = container.task_guard
        self._results_lock = container.results_lock
        self._selector = container.selector
        self._tiered_router = container.tiered_router
        self._adaptive_router = container.adaptive_router
        self._telemetry = container.telemetry
        self._policy_engine = container.policy_engine
        self._job_lock = asyncio.Lock()
        self._project_planner = container.project_planner
        self._pipeline_runner = container.pipeline_runner
        self._hook_registry = container.hook_registry
        self.validator = container.validator
        self._decomposer = container.decomposer
        self._architect = container.architect
        self._executor = container.executor
        self._evaluator = container.evaluator
        self._generator = container.generator
        self._pipeline = container.pipeline
        self._pipeline_executor = container.pipeline_executor
        self._event_bus = container.event_bus
        self._telemetry_store = container.telemetry_store
        self._semantic_cache = container.semantic_cache
        self.observability = getattr(container, "observability", None)
        self._cb_registry = getattr(container, "cb_registry", None)

        # Optimization & Metadata (wired via container or initialized below)
        self.optim_config = getattr(container, "optim_config", None)
        self.meta_v2 = getattr(container, "meta_v2", None)

        container.wire_executor(
            execute_fn=self._execute_task,
            decompose_fn=self._decompose,
        )

        self._run_ctx = build_run_context(budget=self._c.budget)
        self.meta_v2 = build_meta_v2(
            container=container,
            state_manager=self.state_mgr,
        )

        self._project_id: str = ""  # DEPRECATED: use self._run_ctx.project_id
        self.results: dict[str, TaskResult] = (
            self._run_ctx.results
        )  # shared ref to _run_ctx.results
        self._run_ctx.max_parallel_tasks = max(1, max_parallel_tasks)
        self._run_ctx.analyze_on_complete = False
        self._consecutive_failures: dict[Model, int] = dict.fromkeys(Model, 0)
        self._run_ctx.active_policies = PolicySet()
        self.context_truncation_limit: int = 40000
        self._metrics_exporter: MetricsExporter | None = None
        self._run_ctx.channels = {}
        self._entered: bool = False
        self._dashboard_integration: Any | None = None
        self._architecture_rules: Any | None = None
        self._git_integration: Any | None = None
        self._autonomy = AutonomyConfig.for_level(AutonomyLevel.STANDARD)
        self._active_profiles_cache: list | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._cleanup_timer: asyncio.Task | None = None
        self.api_health: dict[Model, bool] = dict.fromkeys(Model, True)
        for model in Model:
            if not self.client.is_available(model):
                self.api_health[model] = False
                logger.warning(f"{model.value}: provider SDK/key not available")
        # P3-2: ModelHealthTracker wraps circuit breaker + telemetry recording.
        # Dicts are shared by reference so engine.api_health and _consecutive_failures
        # stay in sync with what ModelHealthTracker writes.
        # P3-5: Thin bridges for optional dashboard / git integrations.
        # Created before health_tracker so the bridge can be passed in.
        self._dashboard_bridge = build_dashboard_bridge(self._dashboard_integration)
        self._git_bridge = build_git_bridge(self._git_integration)
        # M6: ModelHealthTracker now owns its dicts; pass existing state as
        # initial values so persisted circuit-breaker counts are preserved.
        self._health_tracker = build_health_tracker(
            container=container,
            telemetry=self._telemetry,
            dashboard=self._dashboard_bridge,
            adaptive_router=self._adaptive_router,
            state_mgr=self.state_mgr,
            circuit_breaker_threshold=self._CIRCUIT_BREAKER_THRESHOLD,
            initial_consecutive_failures=self._consecutive_failures,
            initial_api_health=self.api_health,
        )
        # P3-3: ResumptionService wraps _resume_project logic.
        self._resumption_svc = build_resumption_service(
            budget=self.budget,
            results=self.results,
            execute_task_fn=self._execute_task,
            determine_final_status_fn=self._determine_final_status,
        )
        # M3: ProjectRunner wired via callables + run_state (no host back-ref).
        from .application.project_runner import ProjectRunner as _ProjectRunner
        from .application.project_runner_deps import ProjectRunState as _RunState

        self._run_state = _RunState(results=self.results)
        _callables = build_project_runner_callables(
            topological_sort=self._topological_sort,
            topological_levels=self._topological_levels,
            make_state=self._make_state,
            determine_final_status=self._determine_final_status,
            log_summary=self._log_summary,
            execute_all=self._execute_all,
            generate_architecture_rules=self._generate_architecture_rules,
            analyze_completed_project=self._analyze_completed_project,
            client=self._c.client,
            warm_start_fn=self._apply_warm_start,
            flush_telemetry_fn=self._flush_telemetry_snapshots,
            constitution_gate=getattr(self._c, "constitution_gate", None),
        )
        self._project_runner = _ProjectRunner(
            callables=_callables,
            run_state=self._run_state,
            state_mgr=self.state_mgr,
            budget=self.budget,
            event_bus=self._event_bus,
            resumption_svc=self._resumption_svc,
            dashboard_bridge=self._dashboard_bridge,
            git_bridge=self._git_bridge,
            generator=self._generator,
            meta_v2=self.meta_v2,
            cache=self.cache,
            api_health=self.api_health,
            budget_hierarchy=self._c.budget_hierarchy,
        )
        # SkillOpt: self-improving per-TaskType skill documents (P3-4 addendum)
        from .crosscutting.config import flags as _flags

        self._skill_manager = build_skill_manager(container=container, flags=_flags)

        if tracing_cfg is not None and configure_tracing is not None:
            configure_tracing(tracing_cfg)

        # taste-skill: anti-slop design prefix for frontend tasks
        from .crosscutting.config import settings as _settings

        self._taste_skill_service = build_taste_skill_service(flags=_flags, settings=_settings)

        # Phase 5: Late-bind engine-level deps into PipelineExecutor
        container.wire_pipeline_executor(
            skill_manager=getattr(self, "_skill_manager", None),
            taste_skill_service=self._taste_skill_service,
            background_tasks=self._background_tasks,
        )

        logger.info("Orchestrator initialized via ServiceContainer")

    # ─────────────────────────────────────────
    # Accessory Services (Lazy Properties)
    # ─────────────────────────────────────────

    # ── Lazy-init properties removed (Cluster 5) — access via self._c.X directly ──

    # ─────────────────────────────────────────
    # Health Check (P2-3)
    # ─────────────────────────────────────────

    def assert_healthy(self) -> None:
        """Raise RuntimeError if any required service is missing; log optional ones.

        Called automatically in ``__aenter__``.  Can also be called by tests to
        verify that a freshly-built container is wired correctly.
        """
        # Services that are unconditionally required for any task execution
        required = [
            ("budget", self.budget),
            ("client", self.client),
            ("cache", self.cache),
            ("state_mgr", self.state_mgr),
            ("selector", self._selector),
            ("telemetry", self._telemetry),
            ("pipeline", self._pipeline),
            ("validator", self.validator),
            ("task_guard", self._task_guard),
        ]
        # Optional services that are expected in production but degrade gracefully
        optional = [
            ("telemetry_store", self._telemetry_store),
            ("event_bus", self._event_bus),
            ("hook_registry", self._hook_registry),
            ("decomposer", self._decomposer),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            raise ConfigurationError(
                f"Orchestrator missing required services: {missing}",
                details={"missing_services": missing},
            )
        none_optional = [name for name, val in optional if val is None]
        if none_optional:
            logger.info("Optional services not configured: %s", none_optional)

    # ─────────────────────────────────────────
    # Async Context Manager
    # ─────────────────────────────────────────

    async def __aenter__(self) -> Orchestrator:
        """
        Enter async context manager.

        Ensures all resources are properly initialized and will be cleaned up
        on exit. Use this pattern for guaranteed resource cleanup:

            async with Orchestrator() as orch:
                result = await orch.run_project(...)

        Returns:
            Self for use in async with statement

        P0-2 OPTIMIZATION: Starts periodic cleanup timer for background tasks.
        """
        self._entered = True
        if hasattr(self, "_run_state"):
            self._run_state.entered = True
        logger.debug("Orchestrator entered as context manager")

        # Verify all required services are present (P2-3)
        self.assert_healthy()

        # Restore circuit breaker state from previous run (P1-4)
        await self._load_circuit_breaker_state()

        # Start periodic cleanup timer for background tasks
        await self._start_periodic_cleanup(interval_seconds=300)  # 5 minutes

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager, delegating all cleanup to ``_cleanup_resources()``.
        Exceptions during cleanup are logged but not raised to avoid masking
        the original exception.
        """
        await self._cleanup_resources()

    async def _cleanup_resources(self) -> None:
        """
        Release all resources: lifecycle, tasks, cache, state, audit, telemetry.

        Delegates container-managed services to ``self._container.shutdown()``,
        then cleans up Orchestrator-specific state (timers, background tasks, audit).
        """
        logger.debug("Orchestrator cleaning up resources...")

        # ── Phase 1: Orchestrator-specific cleanup ───────────────────────────
        # 1a. Cancel periodic cleanup timer (in snapshotter)
        snap = self._get_snapshotter()
        snap.stop_periodic_cleanup()

        # 1b. Drain background tasks BEFORE container shutdown to avoid
        #     writing to closed telemetry store
        await self._cleanup_background_tasks()
        background_list = list(self._background_tasks)
        if background_list:
            done, pending = await asyncio.wait(background_list, timeout=5.0)
            for task in pending:
                task.cancel()

        # ── Phase 2: Container-managed services ──────────────────────────────
        # Use getattr: a minimally-constructed Orchestrator (or one whose __init__
        # bailed early) may never have set _container, and __aexit__ must not raise.
        if getattr(self, "_c", None) is not None:
            await self._c.shutdown()

        # 2a. Flush telemetry store. Covers stores set directly on the
        #     orchestrator (not only the container-owned one). Idempotent flush,
        #     so no harm if the container already flushed the same object.
        try:
            store = getattr(self, "_telemetry_store", None)
            if store is not None and hasattr(store, "flush"):
                await store.flush()
        except Exception as e:
            logger.warning("Failed to flush telemetry store: %s", e)

        # 2b. Flush audit log (getattr: minimal Orchestrator may lack _audit_log).
        try:
            audit = getattr(self, "_audit_log", None)
            if hasattr(audit, "flush"):
                await audit.flush()
        except Exception as e:
            logger.warning("Failed to flush audit log: %s", e)

        # 2b. Close SkillOpt manager
        try:
            if self._skill_manager is not None:
                await self._skill_manager.close()
        except Exception as e:
            logger.warning("Failed to close skill_manager: %s", e)

        self._entered = False
        if hasattr(self, "_run_state"):
            self._run_state.entered = False

    async def close(self) -> None:
        """
        Explicitly close all resources.

        Called automatically when using async context manager (async with),
        but can also be called explicitly for manual resource management.
        """
        await self.__aexit__(None, None, None)

    # (P3-6: _get_openrouter_call_params and _record_optimization_metrics were
    # confirmed dead code — never called — and removed in this refactoring.)

    # ─────────────────────────────────────────
    # Persistent learning helpers
    # ─────────────────────────────────────────

    async def _apply_warm_start(self) -> None:
        """
        Blend historical ModelProfile data into the in-memory defaults.

        Call this before execution so routing decisions benefit from every
        prior run.  Blending ratios (per plan learn-and-show-design.md):
          COLD (<10 calls):  ignore — keep defaults
          WARM (10-49):      40% historical / 60% default (quality + trust)
          HOT  (≥50):        100% historical (quality, trust, latency)

        Latency is only overridden at HOT confidence to avoid noise.
        """
        from .models import TaskType as _TT

        for model, profile in self._c.planner._profiles.items():
            # Use CODE_GEN as the representative task type for global quality blending.
            # In future, per-task-type blending can be added here.
            hist = await self._telemetry_store.load_historical_profile(model, _TT.CODE_GEN)
            if hist is None:
                continue  # cold start — keep defaults

            if hist.call_count >= 50:
                # HOT: 100% historical
                profile.quality_score = hist.quality_score
                profile.trust_factor = hist.trust_factor
                profile.avg_latency_ms = hist.avg_latency_ms
                profile.latency_p95_ms = hist.latency_p95_ms
            else:
                # WARM: 40% historical / 60% default blend
                profile.quality_score = 0.4 * hist.quality_score + 0.6 * profile.quality_score
                profile.trust_factor = 0.4 * hist.trust_factor + 0.6 * profile.trust_factor

    # ── Telemetry & cleanup — delegated to TelemetrySnapshotter (extracted) ──

    def _get_snapshotter(self):
        """Lazy-init TelemetrySnapshotter — prefer from container."""
        if not hasattr(self, "_snapshotter") or self._snapshotter is None:
            self._snapshotter = getattr(self._c, "snapshotter", None)
            if self._snapshotter is None:
                from .infrastructure.telemetry_snapshotter import TelemetrySnapshotter

                self._snapshotter = TelemetrySnapshotter(
                    telemetry_store=self._c.telemetry_store,
                    get_active_profiles_fn=lambda: [
                        p
                        for p in self._c.planner._profiles.values()
                        if getattr(p, "call_count", 0) >= 1
                    ],
                    background_tasks=self._background_tasks,
                )
        return self._snapshotter

    async def _flush_telemetry_snapshots(self, project_id: str) -> None:
        """Snapshot active model profiles — delegates to TelemetrySnapshotter."""
        await self._get_snapshotter().flush_snapshots(project_id)

    async def _cleanup_background_tasks(self) -> int:
        """Remove completed tasks — delegates to TelemetrySnapshotter."""
        return await self._get_snapshotter().cleanup_done_tasks()

    async def _start_periodic_cleanup(self, interval_seconds: int = 300) -> None:
        """Start periodic cleanup timer — delegates to TelemetrySnapshotter."""
        self._get_snapshotter().start_periodic_cleanup(interval_seconds)

    async def _load_circuit_breaker_state(self) -> None:
        """Restore circuit breaker failure counts from the previous run (P1-4).

        Delegates to ``ModelHealthTracker.load_state()`` and syncs engine-level
        dicts for backward compatibility with ``_make_state`` and decomposition.
        """
        if hasattr(self, "_health_tracker") and self._health_tracker is not None:
            await self._health_tracker.load_state()
            self._consecutive_failures.update(self._health_tracker.consecutive_failures)
            self.api_health.update(self._health_tracker.api_health)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
        app_profile: AppProfile | None = None,  # noqa: F821
        analyze_on_complete: bool = False,
        output_dir: Path | None = None,  # noqa: F821
    ) -> ProjectState:
        """
        Main entry point. Decomposes project → executes tasks → returns state.

        P3-4: Delegates to ProjectRunner. All coordination logic lives there;
        this shell preserves the public API signature and docstring.
        """
        validate_project_args(project_description, success_criteria, project_id, output_dir)
        # Reset per-run state for this new project
        self._run_ctx.reset(
            project_id=project_id,
            budget=self._run_ctx.budget,
            analyze_on_complete=analyze_on_complete,
        )
        return await self._project_runner.run_project(
            project_description=project_description,
            success_criteria=success_criteria,
            project_id=project_id,
            app_profile=app_profile,
            analyze_on_complete=analyze_on_complete,
            output_dir=output_dir,
        )

    async def run_project_with_tasks(
        self,
        project_description: str,
        success_criteria: str,
        tasks: dict,
        project_id: str = "",
        app_profile: Any = None,
        analyze_on_complete: bool = False,
        output_dir: Path | None = None,
        constitution: Any = None,
    ) -> ProjectState:
        """
        Run project with **pre-composed tasks** (skips LLM decomposition).

        Used by ``--from-speckit`` and other ingest paths where tasks
        were already parsed from external artifacts (Spec-Kit, etc.)
        instead of being generated from a raw prompt.

        All other pipeline phases (generate → critique → revise → evaluate)
        run identically to ``run_project()``.
        """
        validate_project_args(project_description, success_criteria, project_id, output_dir)
        # Reset per-run state for this new project
        self._run_ctx.reset(
            project_id=project_id,
            budget=self._run_ctx.budget,
            analyze_on_complete=analyze_on_complete,
        )
        return await self._project_runner.run_project(
            project_description=project_description,
            success_criteria=success_criteria,
            project_id=project_id,
            app_profile=app_profile,
            analyze_on_complete=analyze_on_complete,
            output_dir=output_dir,
            precomposed_tasks=tasks,
            constitution=constitution,
        )

    async def run_job(self, spec: JobSpec) -> ProjectState:
        """
        Policy-driven entry point. Accepts a JobSpec that bundles project
        description, success criteria, budget, quality targets, and policies.

        The active PolicySet is threaded through model selection so that
        ConstraintPlanner enforces compliance on every API call.

        State mutation (budget, policies, parallelism) happens here; the
        lifecycle (warm-start → preflight → run_project → charge → flush)
        is delegated to ProjectRunner.run_job().
        """
        validate_job_spec(spec)
        async with self._job_lock:
            self._run_ctx.budget = spec.budget
            self._run_ctx.active_policies = spec.policy_set
            self._run_ctx.quality_mode = getattr(spec, "quality_mode", "standard")
            # JobSpec may override the per-task parallelism limit
            if spec.max_parallel_tasks > 0:
                self._run_ctx.max_parallel_tasks = spec.max_parallel_tasks
            # Delegate lifecycle to ProjectRunner
            return await self._project_runner.run_job(spec)

    async def run_project_streaming(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
    ):
        """
        Streaming variant of run_project().
        Yields StreamEvent objects as execution progresses.
        The final event is always ProjectCompleted.
        """
        from .streaming import ProjectEventBus

        self._event_bus = ProjectEventBus()
        subscription = self._event_bus.subscribe()

        async def _run() -> None:
            bus = self._event_bus
            try:
                await self.run_project(project_description, success_criteria, project_id)
            finally:
                await bus.close()
                if self._event_bus is bus:
                    self._event_bus = None

        task = asyncio.create_task(_run())

        async for event in subscription:
            yield event

        await task  # propagate any unhandled exceptions

    async def dry_run(
        self, project_description: str, success_criteria: str
    ) -> ExecutionPlan:  # noqa: F821
        """
        Dry-run: decompose the project, build an execution plan, and return it
        WITHOUT executing any tasks. (Improvement 12)

        P3-4: Delegates to ProjectRunner.
        """
        return await self._project_runner.dry_run(
            project_description=project_description,
            success_criteria=success_criteria,
        )

    # ─────────────────────────────────────────
    # Phase 1: Decomposition
    # ─────────────────────────────────────────

    async def _decompose(
        self,
        project: str,
        criteria: str,
        app_profile: AppProfile | None = None,  # noqa: F821
        policy: ResiliencePolicy | None = None,
        project_context: Any = None,  # ProjectContext from architecture rules
    ) -> dict[str, Task]:
        """Delegate to ``decomposer_service.decompose_project``.

        Extracted to ``application/decomposer_service.py`` for testability.
        """
        from .application.decomposer_service import decompose_project

        async def _record_fail(m: Model, error: Exception | None = None) -> None:
            await self._record_failure(m, error=error)

        model = self._select_decomposition_model(project)

        return await decompose_project(
            project=project,
            criteria=criteria,
            model=model,
            client=self.client,
            decomposer=self._decomposer,
            api_health=self.api_health,
            record_failure_fn=_record_fail,
            charge_fn=lambda amount: self.budget.charge(amount, "decomposition"),
            app_profile=app_profile,
            policy=policy,
            project_context=project_context,
        )

    # ─────────────────────────────────────────

    def _check_phase_budget(self, phase: str) -> None:
        """Check phase budget caps via BudgetEnforcer."""
        from .application.budget_enforcer import BudgetEnforcer

        BudgetEnforcer.check_phase_cap(self.budget, phase, logger, self._hook_registry)

    async def _execute_all(
        self,
        tasks: dict[str, Task],
        execution_order: list[str],
        project_desc: str,
        success_criteria: str,
        output_dir: "Path | None" = None,  # noqa: F821
    ) -> ProjectState:
        """
        Execute all tasks respecting dependencies, with intra-level parallelism.
        Delegates to PipelineRunner (engine_core).
        """
        return await self._pipeline_runner.execute_all(
            tasks=tasks,
            execution_order=execution_order,
            project_desc=project_desc,
            success_criteria=success_criteria,
            output_dir=output_dir,
            execute_task_fn=self._execute_task,
            make_state_fn=self._make_state,
        )

    async def _warm_cache_for_level(self, tasks: Dict[str, Task], runnable: List[str]) -> None:
        """Delegate cache warming to ``cache_warmup.warm_cache_for_level``.

        Extracted to ``application/cache_warmup.py`` for testability.
        """
        from .application.cache_warmup import warm_cache_for_level

        await warm_cache_for_level(
            context_service=self._c.context_service,
            results=self.results,
            client=self.client,
        )

    def _validate_syntax_streaming(self, partial_output: str) -> bool:
        """
        Quick streaming syntax validator for early abort.
        Delegates to TaskValidator (engine_core).
        """
        return self.validator.validate_syntax_streaming(partial_output)

    async def _validate_syntax_batch(self, output: str) -> bool:
        """
        Batch syntax validator for post-generation.
        Delegates to TaskValidator (engine_core).
        """
        return self.validator.validate_syntax_batch(output)

    async def _run_preflight_check(
        self,
        task: Task,
        output: str,
        score: float,
        primary: Model,
        policy: ResiliencePolicy | None = None,
    ) -> tuple[str, float, Any]:
        """
        Post-loop preflight delivery gate.
        Delegates to TaskValidator (engine_core).
        """
        return await self.validator.run_preflight_check(
            task=task,
            output=output,
            score=score,
            primary=primary,
            revision_prompt_builder=RevisionPrompt,
            policy=policy,
        )

    def _filter_validators_for_task(self, task: Task, output: str) -> list[str]:
        """
        Filter validators based on task type and content.
        Delegates to TaskValidator (engine_core).
        """
        return self.validator.filter_validators_for_task(task, output)

    async def _execute_task(self, task: Task, policy: ResiliencePolicy | None = None) -> TaskResult:
        """
        Execute a single task via the TaskPipeline.
        Delegates to PipelineExecutor (engine_core).
        """
        return await self._pipeline_executor.execute(task, policy)

    async def _evaluate(self, task: Task, output: str) -> float:
        """Evaluate task quality via EvaluatorService (wired through container)."""
        return await self._evaluator.evaluate(
            task_id=task.id,
            result=output,
            model=task.model if hasattr(task, "model") else None,
        )

    async def _record_success(self, model: Model, response: APIResponse) -> None:
        """Record a successful API call — delegates to ModelHealthTracker (P3-2)."""
        await self._health_tracker.record_success(model, response)
        # Feed rate-limit tracker so _apply_filters can enforce sliding-window caps.
        # Kept here because it requires self._planner which is engine-specific.
        try:
            self._project_planner.rate_limit_tracker.record(
                provider=get_provider(model),
                cost_usd=response.cost_usd,
                tokens=response.input_tokens + response.output_tokens,
            )
        except Exception as _e:
            logger.debug("Rate-limiter record skipped: %s", _e)

    async def _record_failure(self, model: Model, error: Exception | None = None) -> None:
        """Record a failed API call — delegates to ModelHealthTracker (P3-2)."""
        await self._health_tracker.record_failure(model, error)

    def _get_active_policies(self, task_id: str = "") -> list[Policy]:
        """Return merged global + node-level policies for the given task."""
        return self._active_policies.policies_for(task_id)

    def _should_exit_early(
        self,
        scores_history: list[float],
        threshold: float,
        confidence_window: int = 2,
        variance_tolerance: float = 0.001,
    ) -> bool:
        """Determine if execution should exit early — delegates to BudgetEnforcer."""
        from .application.budget_enforcer import BudgetEnforcer

        return BudgetEnforcer.should_exit_early(
            scores_history, threshold, confidence_window, variance_tolerance
        )

    # ─────────────────────────────────────────
    # Model selection & fallback
    # ─────────────────────────────────────────

    # OPTIMIZATION: Tiered model selection for cost efficiency v3.0
    # PRIORITY: Best value models first (Xiaomi, StepFun, GLM, Grok)
    # PHASE-2: Tier constants and escalation count moved to model_selector.TieredModelRouter

    def _get_available_models(self, task_type: TaskType) -> list[Model]:
        """Delegates to TieredModelRouter.available_models."""
        return self._tiered_router.available_models(task_type)

    def _select_decomposition_model(self, project_description: str) -> Model:
        """Delegates to ModelSelector — see model_selector.py for full logic."""
        return self._selector.decomposition_model(project_description)

    # ─────────────────────────────────────────
    # DAG & dependency management
    # ─────────────────────────────────────────

    def _topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """
        Deterministic topological sort.
        Delegates to ProjectPlanner (engine_core).
        """
        return self._project_planner.get_execution_order(tasks)

    def _topological_levels(self, tasks: dict[str, Task]) -> list[list[str]]:
        """
        Group tasks into execution levels.
        Delegates to ProjectPlanner (engine_core).
        """
        return self._project_planner.get_execution_levels(tasks)

    def _filter_validators_for_task(self, task: Task, output: str) -> list[str]:
        """PHASE-4: Delegates to orchestrator.validators.filter_validators_for_task."""
        from .validators import filter_validators_for_task

        return filter_validators_for_task(task, output)

    # (P3-1: _gather_dependency_context was confirmed dead code — never called —
    # and removed in this refactoring. Context injection is handled elsewhere.)

    # ─────────────────────────────────────────
    # Status & resume
    # ─────────────────────────────────────────

    def _determine_final_status(self, state: ProjectState) -> ProjectStatus:
        """Delegates to StateCoordinator."""
        return self._c.state_coordinator.determine_final_status(state)

    async def _resume_project(self, state: ProjectState) -> ProjectState:
        """Resume from last checkpoint — delegates to ResumptionService (P3-3)."""
        return await self._resumption_svc.resume(state)

    def _make_state(
        self,
        project_desc: str,
        criteria: str,
        tasks: dict[str, Task],
        status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS,
        execution_order: list[str] | None = None,
        results: dict[str, TaskResult] | None = None,
    ) -> ProjectState:
        """Delegates to StateCoordinator."""
        return self._c.state_coordinator.make_state(
            project_desc=project_desc,
            criteria=criteria,
            budget=self.budget,
            tasks=tasks,
            results=results if results is not None else dict(self.results),
            api_health={m.value: h for m, h in self.api_health.items()},
            status=status,
            execution_order=execution_order,
        )

    def _log_summary(self, state: ProjectState):
        """Delegates to StateCoordinator."""
        self._c.state_coordinator.log_summary(state)

    async def _analyze_completed_project(self, state: ProjectState, output_dir: Path):  # noqa: F821
        """
        Analyze completed project and generate improvement suggestions.
        Delegates to ProjectAnalyzer — extracted output formatting lives there.
        """
        try:
            from .project_analyzer import ProjectAnalyzer

            logger.info("Running post-project analysis...")
            analyzer = ProjectAnalyzer()
            report = await analyzer.analyze_project(
                project_path=output_dir, project_id=state.project_id, run_quality_gate=True
            )
            summary = analyzer.generate_summary(report)
            logger.info("\n" + summary)
            analyzer.save_report(report, output_dir)
            analyzer.print_suggestions(report)
        except Exception as e:
            logger.warning(f"Project analysis failed: {e}")

    async def _generate_architecture_rules(
        self, project_description: str, success_criteria: str, output_dir: Path | None  # noqa: F821
    ) -> Any | None:
        """
        Generate architecture rules at project start.
        Delegates to Architect service (engine_core).
        """
        return await self._architect.generate_rules(
            project_description=project_description,
            success_criteria=success_criteria,
            output_dir=output_dir,
        )
