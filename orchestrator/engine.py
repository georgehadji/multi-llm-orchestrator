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
import hashlib
import json
import logging
import re
import sqlite3
import time
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .api_clients import APIResponse, UnifiedClient
from .model_selector import ModelSelector
from .task_factory import TaskFactory
from .prompt_builder import (
    CritiquePrompt,
    DecompositionPrompt,
    DeltaPrompt,
    RevisionPrompt,
    SystemPrompt,
)
from .autonomy_config import AutonomyConfig, AutonomyLevel
from .budget import Budget
from .model_registry import ModelRegistry
from .cache import DiskCache
from .models import (
    MODEL_MAX_TOKENS,
    ROUTING_TABLE,
    AttemptRecord,
    Model,
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
    build_default_profiles,
    estimate_cost,
    get_provider,
)
from .resilience import ResiliencePolicy, RetryTemplate
from .semantic_cache import SemanticCache
from .validators import all_validators_pass, async_run_validators
from .exceptions import (
    BudgetExceededError,
    ConfigurationError,
    OrchestratorError,
    TruncatedResponseError,
    TaskError,
)
from .tool_guardrails import ToolCallGuardrailController
from .crosscutting.config import flags

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
# explicit and testable.  Each block checks its flag first; if the flag is
# disabled the import is never attempted and the symbols are set to None.
# All flags default to True to preserve existing behaviour; users can opt out
# via environment variables (e.g. ORCH_A2A_ENABLED=false).

if flags.cache_optimizer_enabled:
    try:
        from .cache_optimizer import CacheConfig, CacheOptimizer

        HAS_CACHE_OPTIMIZER = True
    except ImportError:
        HAS_CACHE_OPTIMIZER = False
        CacheOptimizer = None
        CacheConfig = None
else:
    HAS_CACHE_OPTIMIZER = False
    CacheOptimizer = None
    CacheConfig = None

from .policy import JobSpec, ModelProfile, Policy, PolicySet
from .policy_engine import PolicyEngine
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
            warm_prompt_cache,
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
        warm_prompt_cache = None
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
    warm_prompt_cache = None

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
    from .optimization import OptimizationBackend

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
from .output.code_cleaner import clean_code_output as _clean_code_output  # noqa: E402


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
        cache: "CachePort | DiskCache | None" = None,
        state_manager: "StatePort | StateManager | None" = None,
        max_concurrency: int = 3,
        max_parallel_tasks: int = 3,
        budget_hierarchy: BudgetHierarchy | None = None,
        cost_predictor: CostPredictor | None = None,
        tracing_cfg: TracingConfig | None = None,
        telemetry_store: TelemetryStore | None = None,
        profiles: dict | None = None,
        container: ServiceContainer | None = None,
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

        from .meta_integration import initialize_meta_optimization

        self.meta_v2 = initialize_meta_optimization(
            orchestrator=self,
            state_manager=self.state_mgr,
            enable_transfer_learning=True,
            enable_ab_testing=True,
            enable_hitl=True,
            enable_rollout=True,
        )

        self._project_id: str = ""
        self.results: dict[str, TaskResult] = {}
        self._max_parallel_tasks: int = max(1, max_parallel_tasks)
        self._analyze_on_complete: bool = False
        self._consecutive_failures: dict[Model, int] = dict.fromkeys(Model, 0)
        self._active_policies: PolicySet = PolicySet()
        self.context_truncation_limit: int = 40000
        self._metrics_exporter: MetricsExporter | None = None
        self._channels: dict[str, TaskChannel] = {}
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
        from .application.dashboard_bridge import DashboardBridge as _DashboardBridge
        from .application.git_bridge import GitBridge as _GitBridge

        self._dashboard_bridge = _DashboardBridge(self._dashboard_integration)
        self._git_bridge = _GitBridge(self._git_integration)
        # M6: ModelHealthTracker now owns its dicts; pass existing state as
        # initial values so persisted circuit-breaker counts are preserved.
        from .application.model_health_tracker import ModelHealthTracker as _ModelHealthTracker

        self._health_tracker = _ModelHealthTracker(
            telemetry=self._telemetry,
            dashboard=self._dashboard_bridge,
            adaptive_router=self._adaptive_router,
            state_mgr=self.state_mgr,
            circuit_breaker_threshold=self._CIRCUIT_BREAKER_THRESHOLD,
            initial_consecutive_failures=self._consecutive_failures,
            initial_api_health=self.api_health,
        )
        # P3-3: ResumptionService wraps _resume_project logic.
        from .application.resumption_service import ResumptionService as _ResumptionService

        self._resumption_svc = _ResumptionService(
            budget=self.budget,
            results=self.results,
            execute_task_fn=self._execute_task,
            determine_final_status_fn=self._determine_final_status,
        )
        # M3: ProjectRunner wired via callables + run_state (no host back-ref).
        from .application.project_runner import ProjectRunner as _ProjectRunner
        from .application.project_runner_deps import (
            ProjectRunnerCallables as _Callables,
            ProjectRunState as _RunState,
        )

        self._run_state = _RunState(results=self.results)
        _callables = _Callables(
            topological_sort=self._topological_sort,
            topological_levels=self._topological_levels,
            make_state=self._make_state,
            determine_final_status=self._determine_final_status,
            log_summary=self._log_summary,
            execute_all=self._execute_all,
            generate_architecture_rules=self._generate_architecture_rules,
            analyze_completed_project=self._analyze_completed_project,
            client=self._c.client,
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
        )
        # SkillOpt: self-improving per-TaskType skill documents (P3-4 addendum)
        from .crosscutting.config import flags as _flags

        if _flags.skill_optimization_enabled:
            from .application.skill_store import SkillStore as _SkillStore
            from .application.skill_manager import SkillManager as _SkillManager

            _skill_store = _SkillStore()
            self._skill_manager: Any = _SkillManager(
                optimizer_client=self._c.client,
                skill_store=_skill_store,
            )
            logger.info("SkillOpt enabled — skill_manager initialized")
        else:
            self._skill_manager = None

        if tracing_cfg is not None and configure_tracing is not None:
            configure_tracing(tracing_cfg)

        # taste-skill: anti-slop design prefix for frontend tasks
        from .crosscutting.config import settings as _settings
        from .design.taste_skill_service import TasteSkillService as _TasteSkillService

        self._taste_skill_service = _TasteSkillService(
            flags=_flags,
            settings=_settings,
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
        if getattr(self, "_container", None) is not None:
            await self._container.shutdown()

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
        """Lazy-init TelemetrySnapshotter."""
        if not hasattr(self, "_snapshotter") or self._snapshotter is None:
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

    async def _safe_record_routing_event(
        self,
        project_id: str,
        task_id: str,
        task_type: TaskType,
        result: TaskResult,
    ) -> None:
        """Record routing event — delegates to TelemetrySnapshotter."""
        await self._get_snapshotter().record_routing_event(project_id, task_id, task_type, result)

    async def _load_circuit_breaker_state(self) -> None:
        """Restore circuit breaker failure counts from the previous run (P1-4)."""
        try:
            persisted = await self.state_mgr.load_circuit_breaker_state()
        except Exception as exc:
            logger.debug("Could not load circuit breaker state: %s", exc)
            return
        for model_name, count in persisted.items():
            # Map string name back to Model enum; skip unknown names gracefully
            try:
                model = next(m for m in Model if m.value == model_name)
            except StopIteration:
                continue
            self._consecutive_failures[model] = count
            if count >= self._CIRCUIT_BREAKER_THRESHOLD:
                self.api_health[model] = False
                logger.info(
                    "Circuit breaker restored: %s open (%d failures from previous run)",
                    model_name,
                    count,
                )
        if persisted:
            logger.debug("Loaded circuit breaker state for %d models", len(persisted))
        # M6: propagate loaded state into the tracker's own dicts
        if hasattr(self, "_health_tracker") and self._health_tracker is not None:
            self._health_tracker.update_from_persisted_state(
                self._consecutive_failures, self.api_health
            )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def set_optimization_backend(self, backend: OptimizationBackend) -> None:
        """Swap the ConstraintPlanner's optimization strategy at runtime."""
        self._planner.set_backend(backend)

    @property
    def audit_log(self) -> AuditLog:
        """Read-only access to the policy audit log."""
        return self._audit_log

    @property
    def cost_predictor(self) -> CostPredictor | None:
        """Read-only access to the CostPredictor, if one was configured."""
        return self._cost_predictor

    def add_hook(self, event: str, callback) -> None:
        """Register an event hook callback. See orchestrator.hooks.EventType for event names."""
        self._hook_registry.add(event, callback)

    def set_metrics_exporter(self, exporter: MetricsExporter) -> None:
        """Set the MetricsExporter to use when export_metrics() is called."""
        self._metrics_exporter = exporter

    def export_metrics(self) -> None:
        """Export live per-model telemetry stats via the configured MetricsExporter."""
        if self._metrics_exporter is None:
            return
        self._metrics_exporter.export(self._build_metrics_dict())

    def get_channel(self, name: str) -> TaskChannel:
        """Return the named TaskChannel, creating it lazily on first access."""
        if name not in self._channels:
            self._channels[name] = TaskChannel()
        return self._channels[name]

    # ─────────────────────────────────────────
    # Security & Accountability (arXiv:2602.20021)
    # ─────────────────────────────────────────

    @property
    def task_verifier(self) -> TaskVerifier:
        """Access TaskVerifier for task completion verification."""
        return self._c.task_verifier

    @property
    def accountability(self) -> AccountabilityTracker:
        """Access AccountabilityTracker for action attribution."""
        return self._c.accountability

    @property
    def agent_safety(self) -> AgentSafetyMonitor:
        """Access AgentSafetyMonitor for cross-agent safety."""
        return self._c.agent_safety

    @property
    def red_team(self) -> RedTeamFramework:
        """Access RedTeamFramework for stress testing."""
        return self._c.red_team

    # ─────────────────────────────────────────
    # External Projects Integration (RTK, Mnemo Cortex, LiteLLM)
    # ─────────────────────────────────────────

    @property
    def token_optimizer(self) -> TokenOptimizer:
        """Access TokenOptimizer for CLI output filtering."""
        return self._c.token_optimizer

    @property
    def preflight_validator(self) -> PreflightValidator:
        """Access PreflightValidator for response quality control."""
        return self._c.preflight_validator

    @property
    def session_watcher(self) -> SessionWatcher:
        """Access SessionWatcher for conversation capture."""
        return self._c.session_watcher

    @property
    def persona_manager(self) -> PersonaManager:
        """Access PersonaManager for behavior customization."""
        return self._c.persona_manager

    @property
    def memory_manager(self) -> MemoryTierManager:
        """Access MemoryTierManager for multi-tier memory."""
        return self._c.memory_manager

    @property
    def bm25_search(self) -> BM25Search:
        """Access BM25Search for full-text search."""
        return self._c.bm25_search

    @property
    def reranker(self) -> LLMReranker:
        """Access LLMReranker for result re-ranking."""
        return self._c.reranker

    @property
    def a2a_manager(self) -> A2AManager:
        """Access A2AManager for agent-to-agent communication."""
        return self._c.a2a_manager

    # Convenience methods for external integrations

    def optimize_command_output(self, command: str, output: str) -> str:
        """Optimize command output for token efficiency (RTK)."""
        return self._c.token_optimizer.optimize(command, output)

    def preflight_check(
        self,
        response: str,
        context: dict | None = None,
        mode: PreflightMode = PreflightMode.AUTO,
    ) -> Any:
        """Validate response before sending (Mnemo Cortex)."""
        return self._c.preflight_validator.validate(response, context, mode)

    def start_session(self, project_id: str) -> str:
        """Start a new session for conversation capture."""
        return self._c.session_watcher.start_session(project_id)

    def record_interaction(
        self,
        session_id: str,
        task_input: str,
        task_output: str,
        task_type: str,
        **kwargs,
    ) -> str:
        """Record an interaction in a session."""
        return self._c.session_watcher.record_interaction(
            session_id=session_id,
            task_input=task_input,
            task_output=task_output,
            task_type=task_type,
            **kwargs,
        )

    def set_persona(self, project_id: str, mode: PersonaMode) -> None:
        """Set persona mode for a project."""
        self._c.persona_manager.set_persona(project_id, mode)

    def get_persona_settings(self, project_id: str) -> Any:
        """Get persona settings for a project."""
        return self._c.persona_manager.get_persona_settings(project_id)

    async def store_memory(
        self,
        project_id: str,
        content: str,
        memory_type: str = "task",
    ) -> str:
        """Store a memory in the tiered memory system."""
        from .memory_tier import MemoryType

        return await self._c.memory_manager.store(
            project_id=project_id,
            content=content,
            memory_type=MemoryType(memory_type),
        )

    async def retrieve_memories(
        self,
        project_id: str,
        query: str | None = None,
        limit: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
    ) -> list:
        """
        Retrieve memories from the tiered memory system.

        Args:
            project_id: Project to retrieve from
            query: Search query
            limit: Maximum results
            use_hybrid: Use BM25 hybrid search
            use_reranking: Use LLM re-ranking for better quality

        Returns:
            List of memory entries ordered by relevance
        """
        # Retrieve with hybrid search
        memories = await self._c.memory_manager.retrieve(
            project_id=project_id,
            query=query,
            limit=limit * 2 if use_reranking else limit,  # Get more for reranking
            use_hybrid=use_hybrid,
        )

        # Convert to dicts for reranker
        results = [m.to_dict() for m in memories]

        # Apply re-ranking if enabled and we have results
        if use_reranking and query and results:
            reranked = await self._c.reranker.rerank(query, results, top_k=limit)
            # Convert back to memory entries (or return ranked dicts)
            return [r.to_dict() if hasattr(r, "to_dict") else r for r in reranked]

        return memories[:limit]

    async def hybrid_search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
        use_reranking: bool = True,
        use_query_expansion: bool = True,
    ) -> list:
        """
        Perform hybrid search: BM25 + vector (RRF fusion) + optional reranking.

        Uses HybridSearchPipeline with LLM-based query expansion (DeepSeek-Chat).
        Falls back gracefully if any component is unavailable.
        """
        results = await self._c.hybrid_pipeline.search(
            query,
            project_id=project_id,
            top_k=limit,
            use_reranking=use_reranking,
            use_query_expansion=use_query_expansion,
        )
        return [r.to_dict() for r in results]

    def configure_rate_limits(
        self,
        tenant: str,
        model: str,
        tpm: int,
        rpm: int,
    ) -> None:
        """
        Set TPM/RPM rate limits for a specific tenant and model.

        Args:
            tenant: Tenant identifier (e.g. team name, org ID).
            model: Model identifier string (e.g. "deepseek-chat").
            tpm: Maximum tokens per minute for this tenant+model.
            rpm: Maximum requests per minute for this tenant+model.
        """
        self._c.rate_limiter.set_limits(tenant, model, tpm, rpm)

    def configure_session_lifecycle(
        self,
        migration_interval_hours: int = 1,
        llm_model: str = "deepseek/deepseek-v4-flash",
    ) -> None:
        """
        Configure automatic session lifecycle migration.

        Must be called before starting the scheduler via
        ``await self._lifecycle_manager.start()``.  Raises ``RuntimeError``
        if the scheduler is already running.

        Args:
            migration_interval_hours: How often to run HOT/WARM/COLD migration.
            llm_model: Model used for HOT→WARM entry summarization.
        """
        task = self._c.lifecycle_manager._task
        if task is not None and not task.done():
            raise ConfigurationError(
                "configure_session_lifecycle() must be called before starting the scheduler; "
                "call stop() first, then reconfigure.",
                details={"hint": "call stop() before reconfiguring"},
            )
        self._c.lifecycle_manager._interval = migration_interval_hours * 3600
        self._c.lifecycle_manager._model = llm_model

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: list[str],
    ) -> None:
        """Register an agent for A2A communication."""
        card = AgentCard(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
        )
        await self._c.a2a_manager.register_agent(card)

    async def send_task_to_agent(
        self,
        task_id: str,
        target_agent: str,
        message: str,
        context: dict | None = None,
    ) -> Any:
        """Send a task to another agent via A2A."""
        from .a2a_protocol import TaskSendRequest

        request = TaskSendRequest(
            task_id=task_id,
            target_agent=target_agent,
            message=message,
            context=context or {},
        )
        return await self._c.a2a_manager.send_task(request)

    def register_task_expectations(
        self,
        task_id: str,
        expected_files: list[str],
        expected_outputs: list[str] | None = None,
        required_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
    ) -> None:
        """
        Register expected outcomes for a task (call during planning phase).

        This enables post-completion verification to detect task completion
        misrepresentation (a key vulnerability from the "Agents of Chaos" paper).
        """
        self._c.task_verifier.register_expected_outcome(
            task_id=task_id,
            expected_files=expected_files,
            expected_outputs=expected_outputs,
            required_patterns=required_patterns,
            forbidden_patterns=forbidden_patterns,
        )

    async def verify_task_completion(self, task_id: str) -> Any:
        """
        Verify task completion against registered expectations.

        Returns VerificationResult with discrepancies if any.
        """
        return await self._c.task_verifier.verify_completion(task_id)

    def record_action(
        self,
        actor_id: str,
        actor_type: ActorType,
        actor_name: str,
        action_type: ActionType,
        target: str,
        **kwargs,
    ) -> str:
        """
        Record an action for accountability tracking.

        Returns action_id for linking downstream impacts.
        """
        return self._c.accountability.record_action(
            actor_id=actor_id,
            actor_type=actor_type,
            actor_name=actor_name,
            action_type=action_type,
            target=target,
            **kwargs,
        )

    def track_agent_event(
        self,
        agent_id: str,
        event_type: SafetyEventType,
        severity: int,
        description: str,
    ) -> str:
        """
        Report a safety-relevant event from an agent.

        Returns event_id.
        """
        return self._c.agent_safety.report_event(
            agent_id=agent_id,
            event_type=event_type,
            severity=severity,
            description=description,
        )

    def set_dashboard_integration(self, integration: Any) -> None:
        """Set dashboard integration for real-time updates."""
        self._dashboard_integration = integration

    def _notify_dashboard_project_start(self, project_id: str, state: Any):
        """Delegates to DashboardBridge (P3-5)."""
        self._dashboard_bridge.on_project_start(project_id, state, self._architecture_rules)

    def _notify_dashboard_task_start(self, task_id: str, task: Task, model: Model | None):
        """Delegates to DashboardBridge (P3-5)."""
        self._dashboard_bridge.on_task_start(task_id, task, model)

    def _notify_dashboard_task_progress(self, iteration: int, score: float):
        """Delegates to DashboardBridge (P3-5)."""
        self._dashboard_bridge.on_task_progress(iteration, score)

    def _notify_dashboard_task_complete(self, task_id: str, status: str):
        """Delegates to DashboardBridge (P3-5)."""
        self._dashboard_bridge.on_task_complete(task_id, status)

    def _build_metrics_dict(self) -> dict:
        """Build a per-model metrics dict from live ModelProfile data."""
        result: dict = {}
        for model, profile in self._c.planner._profiles.items():
            result[model.value] = {
                "call_count": profile.call_count,
                "failure_count": profile.failure_count,
                "success_rate": profile.success_rate,
                "avg_latency_ms": profile.avg_latency_ms,
                "latency_p95_ms": profile.latency_p95_ms,
                "quality_score": profile.quality_score,
                "trust_factor": profile.trust_factor,
                "avg_cost_usd": profile.avg_cost_usd,
                "validator_fail_count": profile.validator_fail_count,
                "error_rate": self._telemetry.error_rate(model),
            }
        return result

    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
        app_profile: AppProfile | None = None,
        analyze_on_complete: bool = False,
        output_dir: Path | None = None,
    ) -> ProjectState:
        """
        Main entry point. Decomposes project → executes tasks → returns state.

        P3-4: Delegates to ProjectRunner. All coordination logic lives there;
        this shell preserves the public API signature and docstring.
        """
        return await self._project_runner.run_project(
            project_description=project_description,
            success_criteria=success_criteria,
            project_id=project_id,
            app_profile=app_profile,
            analyze_on_complete=analyze_on_complete,
            output_dir=output_dir,
        )

    async def run_job(self, spec: JobSpec) -> ProjectState:
        """
        Policy-driven entry point. Accepts a JobSpec that bundles project
        description, success criteria, budget, quality targets, and policies.

        The active PolicySet is threaded through model selection so that
        ConstraintPlanner enforces compliance on every API call.
        """
        self.budget = spec.budget
        self._active_policies = spec.policy_set
        self._quality_mode: str = getattr(spec, "quality_mode", "standard")
        # JobSpec may override the per-task parallelism limit
        if spec.max_parallel_tasks > 0:
            self._max_parallel_tasks = spec.max_parallel_tasks
        # Warm-start: blend historical profiles before execution
        await self._apply_warm_start()
        # Extract once; both the pre-flight check and charge path need them.
        job_id = getattr(spec, "job_id", "") or ""
        team = getattr(spec, "team", "") or ""
        # BudgetHierarchy pre-flight check via BudgetEnforcer
        from .application.budget_enforcer import BudgetEnforcer

        BudgetEnforcer.enforce_hierarchy_job(
            self._c.budget_hierarchy, job_id, team, spec.budget.max_usd
        )
        try:
            state = await self.run_project(
                project_description=spec.project_description,
                success_criteria=spec.success_criteria,
            )
        except Exception:
            # BUG-001 FIX: release reservation on failure
            if self._c.budget_hierarchy is not None:
                self._c.budget_hierarchy.release_reservation(job_id, team)
            raise
        # Charge actual spend to BudgetHierarchy
        if self._c.budget_hierarchy is not None:
            actual_spend = self.budget.max_usd - self.budget.remaining_usd
            BudgetEnforcer.enforce_hierarchy_job(
                self._c.budget_hierarchy, job_id, team, spec.budget.max_usd, actual_spend
            )
        # Persist telemetry snapshots for all models used this run (fire-and-forget)
        job_id = getattr(spec, "job_id", "") or self._project_id
        await self._flush_telemetry_snapshots(job_id)
        return state

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

    async def dry_run(self, project_description: str, success_criteria: str) -> ExecutionPlan:
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
        app_profile: AppProfile | None = None,
        policy: ResiliencePolicy | None = None,
    ) -> dict[str, Task]:
        """Break project into atomic tasks via Instructor (fast path) or Decomposer."""
        model = self._select_decomposition_model(project)

        # ── Fast path: Instructor structured decomposition ──────────────────
        _INSTRUCTOR_MAX_CHARS = 8_000
        try:
            from .structured_outputs import TaskDecomposer

            if len(project) <= _INSTRUCTOR_MAX_CHARS:
                decomposer = TaskDecomposer(api_client=self.client)
                decomp_model = (
                    "deepseek/deepseek-v4-flash" if "free" in model.value.lower() else model.value
                )
                logger.info("Using Instructor for structured decomposition with %s", decomp_model)
                result = await decomposer.decompose(
                    project_description=project,
                    success_criteria=criteria,
                    model=decomp_model,
                    max_retries=1,
                )
                tasks = {task.id: task for task in result.to_tasks()}
                logger.info("Instructor decomposition succeeded: %d tasks", len(tasks))
                return tasks
        except ImportError:
            logger.warning("Instructor not available, using Decomposer")
        except Exception as e:
            logger.warning(
                "Instructor decomposition failed (%s), using Decomposer", type(e).__name__
            )

        # ── Fallback: Decomposer from engine_core ───────────────────────────
        async def _record_fail(m: Model, error: Exception | None = None) -> None:
            await self._record_failure(m, error=error)

        return await self._decomposer.decompose(
            project=project,
            criteria=criteria,
            app_profile=app_profile,
            policy=policy,
            api_health=self.api_health,
            record_failure_fn=_record_fail,
            charge_fn=lambda amount: self.budget.charge(amount, "decomposition"),
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
        output_dir: Path | None = None,
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
        """
        OPTIMIZATION: Proactively warm cache before parallel execution.

        Prevents cache miss storm when firing parallel requests.
        When multiple tasks start simultaneously, each would normally create
        its own cache entry. By warming up front with a single call, all
        subsequent parallel calls benefit from the shared cache.

        Args:
            tasks: All tasks in the project
            runnable: Task IDs to be executed in this level
        """
        try:
            # Get system prompt and project context
            system_prompt = self._build_system_prompt()
            project_context = self._build_project_context()

            if not system_prompt and not project_context:
                logger.debug("Cache warming skipped: no system prompt or context")
                return

            # Warm cache with a single call
            await warm_prompt_cache(
                system_prompt=system_prompt,
                project_context=project_context,
                client=self.client,
            )
            logger.info("Cache warmed for parallel execution level")
        except Exception as e:
            logger.warning(f"Cache warming failed (non-critical): {e}")

    def _build_system_prompt(self, task_type: str = "") -> str:
        """Build system prompt based on current quality_mode."""
        return self._c.context_service.build_system_prompt(task_type)

    def _build_project_context(self) -> str:
        """Build project context from existing results."""
        return self._c.context_service.build_project_context(self.results)

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
        Delegates to engine_core.pipeline stages (Generate, Critique, Evaluate, etc).
        """
        from .engine_core.pipeline import PipelineContext
        from .models import TaskStatus

        # Select initial model
        model = task.preferred_model
        if not model and hasattr(self, "_selector") and self._selector:
            model = self._selector.select(task.type)

        # Build skill prefix (SkillOpt + taste-skill)
        skill_prefix = await self._build_skill_prefix(task)

        # Optional image-reference visual-context enrichment
        task = await self._enrich_with_visual_context(task)

        ctx = PipelineContext(
            task=task,
            model=model,
            tokens_used={"input": 0, "output": 0},
            skill_prefix=skill_prefix,
        )

        # Loop for self-consistency / ARA retries
        while True:
            ctx = await self._pipeline.run(ctx)
            if ctx.abort_reason not in ("retry_for_quality", "ara_retry"):
                break
            # Reset for next attempt
            ctx.reset_for_retry()

        # Determine final status
        status = TaskStatus.COMPLETED
        if ctx.score < task.acceptance_threshold:
            status = TaskStatus.DEGRADED
        if ctx.abort_reason and ctx.abort_reason.startswith("stage_error"):
            status = TaskStatus.FAILED

        result = ctx.to_task_result(status=status)

        # taste-skill: soft anti-slop check (WARN only, never blocks)
        self._check_anti_slop(task, ctx)

        # SkillOpt: record trajectory for optimizer (fire-and-forget)
        await self._record_trajectory(task, ctx)

        return result

    async def _build_skill_prefix(self, task: Task) -> str:
        """Build combined skill prefix — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        return await enricher.build_prefix(task)

    async def _enrich_with_visual_context(self, task: Task) -> Task:
        """Enrich task with visual context — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        return await enricher.enrich_with_visual_context(task)

    def _check_anti_slop(self, task: Task, ctx: Any) -> None:
        """Soft anti-slop WARN check — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        enricher.check_anti_slop(task, ctx)

    async def _record_trajectory(self, task: Task, ctx: Any) -> None:
        """Record SkillOpt trajectory — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        await enricher.record_trajectory(task, ctx, self._background_tasks)

    def _get_enricher(self):
        """Lazy-init TaskContextEnricher for backward compatibility."""
        if not hasattr(self, "_ctx_enricher") or self._ctx_enricher is None:
            from .engine_core.stages.context_enricher import TaskContextEnricher

            self._ctx_enricher = TaskContextEnricher(
                skill_manager=self._skill_manager,
                taste_skill_service=self._taste_skill_service,
                client=self._c.client,
            )
        return self._ctx_enricher

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
            self._planner.rate_limit_tracker.record(
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
        """PHASE-2: Delegates to TieredModelRouter.available_models."""
        return self._tiered_router.available_models(task_type)

    def _escalate_tier(self, task_type: TaskType) -> None:
        """PHASE-2: Delegates to TieredModelRouter.escalate_tier."""
        self._tiered_router.escalate_tier(task_type)

    def _select_decomposition_model(self, project_description: str) -> Model:
        """Delegates to ModelSelector — see model_selector.py for full logic."""
        return self._selector.decomposition_model(project_description)

    def _get_fast_decomposition_model(self) -> Model:
        """PHASE-2: Delegates to TieredModelRouter.fast_decomposition_model."""
        return self._tiered_router.fast_decomposition_model()

    def _get_cheapest_available(self) -> Model:
        """PHASE-2: Delegates to TieredModelRouter.cheapest_available."""
        return self._tiered_router.cheapest_available()

    def _select_reviewer(self, generator: Model, task_type: TaskType) -> Model | None:
        """Delegates to ModelSelector — see model_selector.py for full logic."""
        return self._selector.reviewer(generator, task_type)

    def _get_fallback(self, failed_model: Model) -> Model | None:
        """Delegates to ModelSelector — see model_selector.py for full logic."""
        return self._selector.fallback(failed_model)

    def _get_next_tier_model(self, current_model: Model, task_type: TaskType) -> Model | None:
        """Delegates to ModelSelector — see model_selector.py for full logic."""
        return self._selector.next_tier(current_model, task_type)

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
    # Protocol Implementations
    # ─────────────────────────────────────────

    @property
    def spent_usd(self) -> float:
        return self.budget.spent_usd

    @property
    def max_usd(self) -> float:
        return self.budget.max_usd

    async def charge(self, amount: float, phase: str) -> None:
        await self.budget.charge(amount, phase)

    def get_available_models(self, task_type: TaskType) -> list[Model]:
        return self._get_available_models(task_type)

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

    async def _analyze_completed_project(self, state: ProjectState, output_dir: Path):
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
        self, project_description: str, success_criteria: str, output_dir: Path | None
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
