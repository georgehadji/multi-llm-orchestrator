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
from .exceptions import OrchestratorError, TruncatedResponseError
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

try:
    from .audit import AuditLog
except (ImportError, TimeoutError):
    AuditLog = None

try:
    from .bm25_search import BM25Search, get_bm25_search
except (ImportError, TimeoutError):
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

try:
    from .memory_tier import MemoryTierManager
except (ImportError, TimeoutError):
    MemoryTierManager = None

try:
    from .persona import PersonaManager, PersonaMode
except (ImportError, TimeoutError):
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

try:
    from .reranker import LLMReranker, get_reranker
except (ImportError, TimeoutError):
    LLMReranker = None
    get_reranker = None

try:
    from .session_lifecycle import SessionLifecycleManager
except (ImportError, TimeoutError):
    SessionLifecycleManager = None

try:
    from .session_watcher import SessionWatcher
except (ImportError, TimeoutError):
    SessionWatcher = None

# Security & Accountability modules from "Agents of Chaos" paper (arXiv:2602.20021)
try:
    from .task_verifier import TaskVerifier
except (ImportError, TimeoutError):
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

try:
    from .token_optimizer import TokenOptimizer
except (ImportError, TimeoutError):
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
        self.optim_config = getattr(container, 'optim_config', None)
        self.meta_v2 = getattr(container, 'meta_v2', None)

        container.wire_executor(
            execute_fn=self._execute_task,
            decompose_fn=self._decompose,
        )

        from .meta_integration import initialize_meta_optimization
        self.meta_v2 = initialize_meta_optimization(
            orchestrator=self, state_manager=self.state_mgr,
            enable_transfer_learning=True, enable_ab_testing=True,
            enable_hitl=True, enable_rollout=True,
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
        logger.info("Orchestrator initialized via ServiceContainer")

    # ─────────────────────────────────────────
    # Accessory Services (Lazy Properties)
    # ─────────────────────────────────────────

    @property
    def _token_optimizer(self) -> Any:
        return self._c.token_optimizer

    @property
    def _session_watcher(self) -> Any:
        return self._c.session_watcher

    @property
    def _persona_manager(self) -> Any:
        return self._c.persona_manager

    @property
    def _a2a_manager(self) -> Any:
        return self._c.a2a_manager

    @property
    def _red_team(self) -> Any:
        return self._c.red_team

    @property
    def _rate_limiter(self) -> Any:
        return self._c.rate_limiter

    @property
    def _lifecycle_manager(self) -> Any:
        return self._c.lifecycle_manager

    @property
    def _memory_manager(self) -> Any:
        return self._c.memory_manager

    @property
    def _bm25_search(self) -> Any:
        return self._c.bm25_search

    @property
    def _reranker(self) -> Any:
        return self._c.reranker

    @property
    def _knowledge_base(self) -> Any:
        return self._c.knowledge_base

    @property
    def _hybrid_pipeline(self) -> Any:
        return self._c.hybrid_pipeline

    @property
    def _query_expander(self) -> Any:
        return self._c.query_expander

    @property
    def _task_verifier(self) -> Any:
        return self._c.task_verifier

    @property
    def _accountability(self) -> Any:
        return self._c.accountability

    @property
    def _agent_safety(self) -> Any:
        return self._c.agent_safety

    @property
    def _tool_guardrails(self) -> Any:
        return self._c.tool_guardrails

    @property
    def _cache_optimizer(self) -> Any:
        return self._c.cache_optimizer

    @property
    def _prompt_cacher(self) -> Any:
        return self._c.prompt_cacher

    @property
    def _budget_hierarchy(self) -> Any:
        """BudgetHierarchy from the container (used by run_job)."""
        return self._c.budget_hierarchy

    @property
    def _batch_client(self) -> Any:
        return self._c.batch_client

    @property
    def _token_budget(self) -> Any:
        return self._c.token_budget

    @property
    def _model_cascader(self) -> Any:
        return self._c.model_cascader

    @property
    def _speculative_gen(self) -> Any:
        return self._c.speculative_gen

    @property
    def _streaming_validator(self) -> Any:
        return self._c.streaming_validator

    @property
    def _dependency_injector(self) -> Any:
        return self._c.dependency_injector

    @property
    def _adaptive_temp(self) -> Any:
        return self._c.adaptive_temp

    @property
    def _tdd_generator(self) -> Any:
        return self._c.tdd_generator

    @property
    def _diff_generator(self) -> Any:
        return self._c.diff_generator

    @property
    def _eval_dataset(self) -> Any:
        return self._c.eval_dataset

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
            raise RuntimeError(f"Orchestrator missing required services: {missing}")
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
        Exit async context manager, ensuring all resources are cleaned up.

        Cleanup order:
        1. Cancel periodic cleanup timer (P0-2 OPTIMIZATION)
        2. Clean up completed background tasks (BUG-MEMORY-002 FIX)
        3. Wait for pending background tasks (BUG-SHUTDOWN-001 FIX)
        4. Flush any pending telemetry snapshots
        5. Close cache connection
        6. Close state manager connection
        7. Flush audit log
        8. Flush telemetry store

        Exceptions during cleanup are logged but not raised to avoid masking
        the original exception.

        BUG-EVENTLOOP-001 FIX: Properly wait for aiosqlite background threads
        to complete before event loop closes.

        """
        logger.debug("Orchestrator exiting context manager, cleaning up resources...")

        # Stop session lifecycle scheduler (no-op if never started)
        try:
            await self._lifecycle_manager.stop()
        except Exception as e:
            logger.warning(f"Failed to stop lifecycle manager: {e}")

        # P0-2 OPTIMIZATION: Cancel periodic cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            try:
                await self._cleanup_timer
            except asyncio.CancelledError:
                pass
            logger.debug("Periodic cleanup timer cancelled")

        # BUG-MEMORY-002 FIX: Clean up completed background tasks first
        await self._cleanup_background_tasks()

        background_list = list(self._background_tasks)
        if background_list:
            logger.debug(f"Waiting for {len(background_list)} background tasks...")
            done, pending = await asyncio.wait(
                background_list,
                timeout=5.0,  # Don't wait forever
                return_when=asyncio.ALL_COMPLETED,
            )
            if pending:
                logger.warning(f"{len(pending)} background tasks did not complete in time")
                # Cancel pending tasks to prevent resource leak
                for task in pending:
                    task.cancel()
            logger.debug(f"Background tasks complete: {len(done)} succeeded")

        # 1. Flush telemetry if we have a project ID
        if self._project_id:
            try:
                await self._flush_telemetry_snapshots(self._project_id)
                logger.debug("Telemetry snapshots flushed")
            except Exception as e:
                logger.warning(f"Failed to flush telemetry snapshots: {e}")

        # 2. Close cache connection with proper shutdown (BUG-EVENTLOOP-001 FIX)
        try:
            await self.cache.close()
            # Yield control to allow aiosqlite background thread to finish
            await asyncio.sleep(0)
            logger.debug("Cache connection closed")
        except Exception as e:
            logger.warning(f"Failed to close cache connection: {e}")

        # 3. Close state manager connection with proper shutdown (BUG-EVENTLOOP-001 FIX)
        try:
            await self.state_mgr.close()
            # Yield control to allow aiosqlite background thread to finish
            await asyncio.sleep(0)
            logger.debug("State manager connection closed")
        except Exception as e:
            logger.warning(f"Failed to close state manager connection: {e}")

        # 4. Flush audit log if needed
        try:
            if hasattr(self._audit_log, "flush"):
                await self._audit_log.flush()
                logger.debug("Audit log flushed")
        except Exception as e:
            logger.warning(f"Failed to flush audit log: {e}")

        # 5. Flush telemetry store
        try:
            if self._telemetry_store is not None:
                await self._telemetry_store.flush()
            logger.debug("Telemetry store flushed")
        except Exception as e:
            logger.warning(f"Failed to flush telemetry store: {e}")

        # 6. SkillOpt cleanup
        try:
            if self._skill_manager is not None:
                await self._skill_manager.close()
        except Exception as e:
            logger.warning("Failed to close skill_manager: %s", e)

        self._entered = False
        if hasattr(self, "_run_state"):
            self._run_state.entered = False
        logger.debug("Orchestrator cleanup complete")

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

        for model, profile in self._profiles.items():
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

    async def _flush_telemetry_snapshots(self, project_id: str) -> None:
        """
        Fire-and-forget: snapshot each ModelProfile that was used this run.
        Only profiles with call_count >= 1 are written.
        Uses asyncio.create_task so the hot path is never blocked.

        BUG-SHUTDOWN-001 FIX: Task is tracked for proper shutdown waiting.
        BUG-MEMORY-002 FIX: Added exception handling in callback to prevent leaks.
        P1-1 OPTIMIZATION: Uses cached active profiles to avoid iterating all models.
        P2-2 OPTIMIZATION: Uses batch insert for 10x faster writes.
        FIX-005a: Handles validation errors and logs detailed failures.
        """

        async def _write_snapshots() -> None:
            # P2-2 OPTIMIZATION: Use batch insert instead of individual calls
            # Collect all active profiles and insert in single transaction
            active_profiles = self._get_active_profiles()
            if active_profiles:
                try:
                    result = await self._telemetry_store.record_snapshots_batch(
                        project_id, active_profiles
                    )
                    # FIX-005a: Handle validation errors
                    if result.get("failed", 0) > 0:
                        logger.warning(
                            f"Telemetry batch: {result['success']} succeeded, "
                            f"{result['failed']} failed"
                        )
                        for err in result.get("errors", [])[:5]:  # Log first 5 errors
                            logger.warning(
                                f"  - {err.get('model', 'unknown')}: {err.get('error', 'unknown')}"
                            )
                    else:
                        logger.debug(
                            f"P2-2: Batch telemetry flush complete for {len(active_profiles)} models"
                        )
                except Exception as exc:
                    logger.warning(f"TelemetryStore.record_snapshots_batch failed: {exc}")
            else:
                logger.debug("P2-2: No active profiles to flush")

        task = asyncio.create_task(_write_snapshots())
        self._background_tasks.add(task)
        task.add_done_callback(self._cleanup_task_callback)

    def _cleanup_task_callback(self, task: asyncio.Task) -> None:
        """Done-callback: remove task from the strong-reference set and log failures."""
        self._background_tasks.discard(task)
        if task.cancelled():
            logger.debug("Background task was cancelled")
        elif task.exception() is not None:
            logger.warning("Background task failed: %s", task.exception())
        else:
            logger.debug("Background task completed successfully")

    async def _cleanup_background_tasks(self) -> int:
        """Remove completed tasks from the tracking set.

        Returns:
            Number of tasks removed.
        """
        if not self._background_tasks:
            return 0
        done = {t for t in self._background_tasks if t.done()}
        self._background_tasks -= done
        logger.debug("Background tasks cleaned up: %d done, %d still running", len(done), len(self._background_tasks))
        return len(done)

    async def _start_periodic_cleanup(self, interval_seconds: int = 300) -> None:
        """Start periodic cleanup timer for completed background tasks.

        Args:
            interval_seconds: How often to run cleanup (default: 5 minutes)
        """

        async def _cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                await self._cleanup_background_tasks()

        self._cleanup_timer = asyncio.create_task(_cleanup_loop())
        logger.info("Started periodic cleanup timer (interval=%ds)", interval_seconds)

    async def _safe_record_routing_event(
        self,
        project_id: str,
        task_id: str,
        task_type: TaskType,
        result: TaskResult,
    ) -> None:
        """Fire-and-forget wrapper: record a routing event, swallowing exceptions."""
        try:
            await self._telemetry_store.record_routing_event(project_id, task_id, task_type, result)
        except Exception as exc:
            logger.warning("TelemetryStore.record_routing_event failed: %s", exc)

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
                    model_name, count,
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
        return self._task_verifier

    @property
    def accountability(self) -> AccountabilityTracker:
        """Access AccountabilityTracker for action attribution."""
        return self._accountability

    @property
    def agent_safety(self) -> AgentSafetyMonitor:
        """Access AgentSafetyMonitor for cross-agent safety."""
        return self._agent_safety

    @property
    def red_team(self) -> RedTeamFramework:
        """Access RedTeamFramework for stress testing."""
        return self._red_team

    # ─────────────────────────────────────────
    # External Projects Integration (RTK, Mnemo Cortex, LiteLLM)
    # ─────────────────────────────────────────

    @property
    def token_optimizer(self) -> TokenOptimizer:
        """Access TokenOptimizer for CLI output filtering."""
        return self._token_optimizer

    @property
    def preflight_validator(self) -> PreflightValidator:
        """Access PreflightValidator for response quality control."""
        return self._preflight_validator

    @property
    def session_watcher(self) -> SessionWatcher:
        """Access SessionWatcher for conversation capture."""
        return self._session_watcher

    @property
    def persona_manager(self) -> PersonaManager:
        """Access PersonaManager for behavior customization."""
        return self._persona_manager

    @property
    def memory_manager(self) -> MemoryTierManager:
        """Access MemoryTierManager for multi-tier memory."""
        return self._memory_manager

    @property
    def bm25_search(self) -> BM25Search:
        """Access BM25Search for full-text search."""
        return self._bm25_search

    @property
    def reranker(self) -> LLMReranker:
        """Access LLMReranker for result re-ranking."""
        return self._reranker

    @property
    def a2a_manager(self) -> A2AManager:
        """Access A2AManager for agent-to-agent communication."""
        return self._a2a_manager

    # Convenience methods for external integrations

    def optimize_command_output(self, command: str, output: str) -> str:
        """Optimize command output for token efficiency (RTK)."""
        return self._token_optimizer.optimize(command, output)

    def preflight_check(
        self,
        response: str,
        context: dict | None = None,
        mode: PreflightMode = PreflightMode.AUTO,
    ) -> Any:
        """Validate response before sending (Mnemo Cortex)."""
        return self._preflight_validator.validate(response, context, mode)

    def start_session(self, project_id: str) -> str:
        """Start a new session for conversation capture."""
        return self._session_watcher.start_session(project_id)

    def record_interaction(
        self,
        session_id: str,
        task_input: str,
        task_output: str,
        task_type: str,
        **kwargs,
    ) -> str:
        """Record an interaction in a session."""
        return self._session_watcher.record_interaction(
            session_id=session_id,
            task_input=task_input,
            task_output=task_output,
            task_type=task_type,
            **kwargs,
        )

    def set_persona(self, project_id: str, mode: PersonaMode) -> None:
        """Set persona mode for a project."""
        self._persona_manager.set_persona(project_id, mode)

    def get_persona_settings(self, project_id: str) -> Any:
        """Get persona settings for a project."""
        return self._persona_manager.get_persona_settings(project_id)

    async def store_memory(
        self,
        project_id: str,
        content: str,
        memory_type: str = "task",
    ) -> str:
        """Store a memory in the tiered memory system."""
        from .memory_tier import MemoryType

        return await self._memory_manager.store(
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
        memories = await self._memory_manager.retrieve(
            project_id=project_id,
            query=query,
            limit=limit * 2 if use_reranking else limit,  # Get more for reranking
            use_hybrid=use_hybrid,
        )

        # Convert to dicts for reranker
        results = [m.to_dict() for m in memories]

        # Apply re-ranking if enabled and we have results
        if use_reranking and query and results:
            reranked = await self._reranker.rerank(query, results, top_k=limit)
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
        results = await self._hybrid_pipeline.search(
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
        self._rate_limiter.set_limits(tenant, model, tpm, rpm)

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
        task = self._lifecycle_manager._task
        if task is not None and not task.done():
            raise RuntimeError(
                "configure_session_lifecycle() must be called before starting the scheduler. "
                "Call stop() first, then reconfigure."
            )
        self._lifecycle_manager._interval = migration_interval_hours * 3600
        self._lifecycle_manager._model = llm_model

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
        await self._a2a_manager.register_agent(card)

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
        return await self._a2a_manager.send_task(request)

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
        self._task_verifier.register_expected_outcome(
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
        return await self._task_verifier.verify_completion(task_id)

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
        return self._accountability.record_action(
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
        return self._agent_safety.report_event(
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
        for model, profile in self._profiles.items():
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
        # BudgetHierarchy pre-flight check (Improvement 6)
        if self._budget_hierarchy is not None:
            if not self._budget_hierarchy.can_afford_job(job_id, team, spec.budget.max_usd):
                raise ValueError(
                    f"BudgetHierarchy rejects job '{job_id}': "
                    "org/team/job limits would be exceeded"
                )
        try:
            state = await self.run_project(
                project_description=spec.project_description,
                success_criteria=spec.success_criteria,
            )
        except Exception:
            # BUG-001 FIX: release the reservation made by can_afford_job() so the
            # org/team budget is not permanently locked when run_project() fails.
            if self._budget_hierarchy is not None:
                self._budget_hierarchy.release_reservation(job_id, team)
            raise
        # Charge actual spend to BudgetHierarchy so cross-run caps are enforced.
        if self._budget_hierarchy is not None:
            actual_spend = self.budget.max_usd - self.budget.remaining_usd
            self._budget_hierarchy.charge_job(job_id, team, actual_spend)
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
        """Use cheapest capable model to break project into atomic tasks."""
        valid_types = [t.value for t in TaskType]

        # Build optional app-context block injected into the prompt
        app_context_block = ""
        if app_profile is not None:
            from orchestrator.scaffold import _TEMPLATE_MAP
            from orchestrator.scaffold.templates import generic

            template_files = _TEMPLATE_MAP.get(app_profile.app_type, generic.FILES)
            scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
            tech_stack_str = (
                ", ".join(app_profile.tech_stack) if app_profile.tech_stack else "unknown"
            )

            # Build architecture block if ArchitectureDecision fields are present
            arch_block = ""
            if hasattr(app_profile, "structural_pattern") and app_profile.structural_pattern:
                rationale_line = (
                    f"\n  Rationale:          {app_profile.rationale}"
                    if getattr(app_profile, "rationale", "")
                    else ""
                )
                arch_block = f"""
ARCHITECTURE DECISION:
  Structural pattern: {app_profile.structural_pattern}
  Topology:           {app_profile.topology}
  API paradigm:       {app_profile.api_paradigm}
  Data paradigm:      {app_profile.data_paradigm}{rationale_line}

Each task MUST follow this architecture — do not invent an alternative structure.
"""

            app_context_block = f"""
APP_TYPE: {app_profile.app_type}
TECH_STACK: {tech_stack_str}
SCAFFOLD_FILES (already exist — fill or extend these):
{scaffold_list}
{arch_block}
Each task JSON element MUST also include:
- "target_path": the relative file path this task writes (e.g. "app/page.tsx").
  Use the exact scaffold paths listed above where applicable.
  Tasks producing non-file outputs (code_review, evaluation) use target_path: "".
- "tech_context": brief note on the tech stack relevant to this specific file.
"""

        prompt = DecompositionPrompt.build(project, criteria, app_context_block, valid_types)

        decomp_system = "You are a precise project decomposition engine. Output only valid JSON."
        # P1-2 OPTIMIZATION: Use adaptive model selection based on project complexity
        model = self._select_decomposition_model(project)

        # Try Instructor for structured decomposition (skip for large projects — Instructor
        # adds ~27K tokens of schema overhead, which overflows 32K-context models).
        _INSTRUCTOR_MAX_CHARS = 8_000  # ~2K tokens; leaves headroom for schema overhead
        try:
            from .structured_outputs import TaskDecomposer

            if len(project) > _INSTRUCTOR_MAX_CHARS:
                raise ValueError(
                    f"Project description too large for Instructor ({len(project)} chars)"
                )

            decomposer = TaskDecomposer(api_client=self.client)
            decomp_model = (
                "deepseek/deepseek-v4-flash"  # cost-effective fallback for free-tier models
                if "free" in model.value.lower()
                else model.value
            )
            logger.info(f"Using Instructor for structured decomposition with {decomp_model}")
            result = await decomposer.decompose(
                project_description=project,
                success_criteria=criteria,
                model=decomp_model,
                max_retries=1,  # fail fast; manual parsing is the reliable fallback
            )
            tasks = {task.id: task for task in result.to_tasks()}
            logger.info(f"Instructor decomposition succeeded: {len(tasks)} tasks")
            return tasks
        except ImportError:
            logger.warning("Instructor not available, falling back to manual JSON parsing")
        except Exception as e:
            import traceback

            # BUG-005: Enhanced logging for Instructor validation failures
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["assert", "validation", "schema", "parse"]):
                logger.warning(f"Instructor validation failed: {type(e).__name__}: {e}")
                logger.debug(f"Instructor failed input preview: {project[:200]}...")
            else:
                logger.warning(f"Instructor decomposition failed: {type(e).__name__}: {e}")
            logger.debug(f"Instructor traceback: {traceback.format_exc()}")
            logger.warning("Falling back to manual JSON parsing")

        # FALLBACK: Original decomposition logic
        last_response_text = ""

        # OpenRouter Optimization: Check if JSON schema responses are enabled
        use_json_schema = (
            OPENROUTER_OPTS is not None
            and OPENROUTER_OPTS.USE_JSON_SCHEMA_RESPONSES
            and generate_openrouter_schema is not None
        )
        if use_json_schema:
            logger.info("OpenRouter: Using JSON schema structured output for decomposition")

        async def _try_decompose(m: Model | str, max_tokens: int = 8192) -> dict[str, Task]:
            nonlocal last_response_text

            # Build call arguments
            call_args = {
                "model": m,
                "prompt": prompt,
                "system": decomp_system,
                "max_tokens": max_tokens,
                "timeout": 120,
                "bypass_cache": True,  # never reuse a cached decomposition response
            }

            # Add OpenRouter optimization parameters if enabled
            # Use REASONING task type for decomposition (better structured output)
            if use_json_schema:
                call_args["task_type"] = TaskType.REASONING
                call_args["response_schema"] = True

            if policy is not None:
                call_args["policy"] = policy

            # TASK-301: routed through DecomposerService (last self.client.call in engine.py)
            class _CompatResponse:
                """Minimal response wrapper to satisfy downstream truncation checks."""
                def __init__(self, text: str): self.text = text
            try:
                resp = await self._decomposer.decompose(
                    description=str(call_args.get("prompt", "")),
                    project_context=str(call_args.get("system", "")),
                )
                resp_text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
            except Exception as _decomp_err:
                logger.warning(f"DecomposerService failed, using direct client: {_decomp_err}")
                resp = await self.client.call(**call_args)
                resp_text = resp.text
            resp = _CompatResponse(resp_text)
            last_response_text = resp.text  # Capture for error logging

            # Check for truncation by examining response text
            # If response ends abruptly without proper JSON closure, it's likely truncated
            text_stripped = resp.text.strip()
            is_truncated = (text_stripped.startswith("[") and not text_stripped.endswith("]")) or (
                len(text_stripped) > 100  # Has content
                and text_stripped.count("{") > text_stripped.count("}")  # Unclosed braces
            )

            if is_truncated:
                logger.warning(
                    f"Detected truncated response ({len(resp.text)} chars, "
                    f"max_tokens={max_tokens})"
                )
                raise TruncatedResponseError(
                    tokens_used=max_tokens,  # We don't know exact usage, assume max
                    max_tokens=max_tokens,
                )

            await self.budget.charge(resp.cost_usd, "decomposition")
            await self._record_success(m if isinstance(m, Model) else model, resp)
            result = self._parse_decomposition(resp.text)
            if not result:
                model_name = m.value if isinstance(m, Model) else m
                raise ValueError(f"Decomposition returned empty task list from {model_name}")
            return result

        # Try primary model, then fallback, with one retry on empty/malformed output
        # v3.0 FIX: If JSON parsing fails, try with a model known for structured output
        models_to_try = [model, self._get_fallback(model)]

        # Add Qwen3 Coder Next as final fallback for JSON structure issues
        if Model.QWEN_3_6_FLASH not in models_to_try:
            models_to_try.append(Model.QWEN_3_6_FLASH)

        # OpenRouter Optimization: Apply model variants if enabled
        # Use REASONING variant (THINKING) for decomposition tasks
        use_model_variants = OPENROUTER_OPTS is not None and OPENROUTER_OPTS.USE_MODEL_VARIANTS
        if use_model_variants:
            from .models import TASK_VARIANT_STRATEGY, ModelVariant

            variant = TASK_VARIANT_STRATEGY.get(TaskType.REASONING, ModelVariant.NONE)
            if variant != ModelVariant.NONE:
                logger.info(f"OpenRouter: Using model variant '{variant.value}' for decomposition")
                # Convert models to variant-suffixed strings
                models_to_try_str = []
                for m in models_to_try:
                    if m is not None:
                        models_to_try_str.append(m.with_variant(variant))
                # Update the list for the loop (will be used in _try_decompose)
                # Store original models for logging
                original_models = models_to_try
                models_to_try = models_to_try_str

        # Track token escalation for truncation recovery
        token_limits = [8192, 12288, 16384]  # Escalating token limits (16K max)
        truncation_attempts = 0

        for attempt, m in enumerate(models_to_try):
            if m is None:
                break
            # Get model name for logging (handle both Model enum and string variants)
            model_name = m.value if isinstance(m, Model) else m

            # Select token limit (escalate on truncation retries)
            current_max_tokens = token_limits[min(truncation_attempts, len(token_limits) - 1)]

            try:
                return await _try_decompose(m, max_tokens=current_max_tokens)
            except TruncatedResponseError as e:
                # Response was truncated - escalate tokens and retry
                truncation_attempts += 1
                if truncation_attempts < len(token_limits):
                    logger.warning(
                        f"Decomposition attempt {attempt + 1} with {model_name} "
                        f"failed (truncated). Retrying with {token_limits[truncation_attempts]} tokens..."
                    )
                    # Retry same model with more tokens (don't count as model failure)
                    try:
                        return await _try_decompose(m, max_tokens=token_limits[truncation_attempts])
                    except TruncatedResponseError:
                        pass  # Continue to next model with even more tokens
                else:
                    logger.error(
                        f"Decomposition with {model_name} still truncated even with "
                        f"{token_limits[-1]} tokens"
                    )
                await self._record_failure(m if isinstance(m, Model) else model, error=e)
            except (json.JSONDecodeError, ValueError) as e:
                # JSON parsing failed - try next model
                raw_preview = last_response_text[:300] if last_response_text else "N/A"
                logger.warning(
                    f"Decomposition attempt {attempt + 1} with {model_name} failed (JSON parse error): {e}"
                )
                logger.warning(f"  Raw response (first 300 chars): {raw_preview}...")
                await self._record_failure(m if isinstance(m, Model) else model, error=e)
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(1)
            except (Exception, asyncio.CancelledError) as e:
                error_type = type(e).__name__
                logger.error(
                    f"Decomposition attempt {attempt + 1} with {model_name} failed "
                    f"({error_type}): {e}"
                )
                await self._record_failure(m if isinstance(m, Model) else model, error=e)
                # Brief pause before the next model — avoids hammering all fallbacks
                # simultaneously during transient network failures (DNS blips, etc.)
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(2)

        logger.error("All decomposition attempts failed")
        raise OrchestratorError(
            "Project decomposition failed after exhausting all model fallbacks. "
            "Check network connectivity and API key validity, then retry. "
            "If the issue persists, simplify the project description or use --resume to continue."
        )

    def _try_parse_partial_json_array(self, text: str) -> list | None:
        """
        Attempt to parse a potentially truncated JSON array.

        When LLM responses get cut off mid-stream, we may have valid JSON objects
        at the start but missing the closing brackets. This method tries multiple
        strategies to recover as much data as possible.

        Args:
            text: Potentially truncated JSON array text

        Returns:
            List of parsed objects, or None if recovery fails
        """
        import re

        text_stripped = text.strip()

        # Quick check: if text is empty or just '[', nothing to recover
        if not text_stripped or text_stripped == "[":
            logger.warning("Response is empty or just '[' - no recoverable content")
            return None

        # Strategy 0: Handle extreme truncation by detecting partial first object
        # If text starts with '[{' but has no complete objects, try to extract
        # whatever fields are present
        if text_stripped.startswith("[{") and "}" not in text_stripped:
            # Try to extract ID if present (minimum viable recovery)
            id_match = re.search(r'"id"\s*:\s*"(task_[^"]*)"', text_stripped)
            if id_match:
                task_id = id_match.group(1)
                logger.warning(f"Extreme truncation detected - only found ID: {task_id}")
                # Create minimal viable task object
                minimal_task = {
                    "id": task_id,
                    "type": "code_generation",
                    "prompt": "Implement the project requirements (auto-generated due to truncation)",
                    "dependencies": [],
                    "acceptance_threshold": 0.8,
                }
                logger.info("Created minimal recovery task from partial ID")
                return [minimal_task]

        # Strategy 1: Try to find complete {...} objects and parse them individually
        # This handles cases where the array is truncated mid-object
        objects = []
        depth = 0
        start_idx = None

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start_idx is not None:
                    # Found a complete object
                    obj_text = text[start_idx : i + 1]
                    try:
                        obj = json.loads(obj_text)
                        if isinstance(obj, dict) and "id" in obj:
                            objects.append(obj)
                    except json.JSONDecodeError:
                        # Try with json5 if available
                        try:
                            import json5

                            obj = json5.loads(obj_text)
                            if isinstance(obj, dict) and "id" in obj:
                                objects.append(obj)
                        except Exception:
                            pass  # Skip malformed/incomplete objects
                    start_idx = None

        if objects:
            logger.info(f"Recovered {len(objects)} complete task objects from truncated response")
            return objects

        # Strategy 2: Try line-by-line parsing (for line-delimited JSON)
        # Some models output JSONL format instead of a JSON array
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "id" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    try:
                        import json5

                        obj = json5.loads(line)
                        if isinstance(obj, dict) and "id" in obj:
                            objects.append(obj)
                    except Exception:
                        pass

        if objects:
            logger.info(f"Recovered {len(objects)} task objects from line-delimited JSON")
            return objects

        # Strategy 3: Try aggressive cleanup and re-parse
        # Remove trailing commas, incomplete final objects, etc.
        cleaned_text = text.strip()
        # Remove leading '[' if present
        if cleaned_text.startswith("["):
            cleaned_text = cleaned_text[1:]
        # Try to find the last complete object
        last_brace = cleaned_text.rfind("}")
        if last_brace > 0:
            # Truncate to last complete object
            truncated = cleaned_text[: last_brace + 1]
            # Try to wrap in array brackets
            try:
                wrapped = "[" + truncated + "]"
                items = json.loads(wrapped)
                if isinstance(items, list) and items:
                    logger.info(f"Recovered {len(items)} task objects via aggressive cleanup")
                    return items
            except json.JSONDecodeError:
                pass

        # Strategy 4: Look for "id": "task_" patterns and try to extract objects around them
        # This is a last resort for badly malformed responses
        task_matches = list(re.finditer(r'"id"\s*:\s*"task_[^"]*"', text))
        if task_matches:
            logger.warning(
                f"Found {len(task_matches)} task IDs but couldn't parse full objects. "
                "Response may be too truncated to recover."
            )
            # Try to extract objects by looking backwards from each task ID
            for match in task_matches:
                # Look for opening brace before this id
                start = text.rfind("{", 0, match.start())
                if start >= 0:
                    # Look for closing brace after the id
                    end = text.find("}", match.end())
                    if end > 0:
                        obj_text = text[start : end + 1]
                        try:
                            obj = json.loads(obj_text)
                            if isinstance(obj, dict) and "id" in obj:
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass

            if objects:
                logger.info(f"Recovered {len(objects)} task objects via pattern extraction")
                return objects

        return None

    def _parse_decomposition(self, text: str) -> dict[str, Task]:
        """
        Parse LLM output into Task objects with defensive handling.

        P2-1 OPTIMIZATION: Uses json5 library for robust JSON parsing that handles:
        - Trailing commas
        - Single-quoted strings
        - Comments
        - Unquoted keys

        This replaces the multi-pass regex approach with a single robust parse.
        """
        text = text.strip()

        # Strip markdown fences (``` or ```json)
        if text.startswith("```"):
            text = re.sub(r"^```\w*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        # P2-1 OPTIMIZATION: Try json5 first (handles trailing commas, comments, etc.)
        items = None
        try:
            import json5

            items = json5.loads(text)
            logger.debug("P2-1: JSON5 parse succeeded on full text")
        except ImportError:
            # json5 not installed, fall back to standard json
            logger.debug("P2-1: json5 not available, using standard json")
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                items = None
        except Exception as e:
            # JSON5 parse failed, try fallback strategies
            logger.debug(f"P2-1: JSON5 parse failed ({e}), trying fallback strategies")
            items = None

        # Fallback: If json5 failed or not available, try standard json with fixes
        if items is None:

            def _try_parse_standard(s: str):
                """Attempt json.loads with progressively more aggressive fixes."""
                # 1. Direct parse
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
                # 2. Strip trailing commas before ] or } (common LLM mistake)
                cleaned = re.sub(r",\s*([}\]])", r"\1", s)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                # 3. Remove control characters (except \n \r \t)
                cleaned2 = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
                try:
                    return json.loads(cleaned2)
                except json.JSONDecodeError:
                    pass
                return None

            items = _try_parse_standard(text)

        # If the top-level is a dict, look for a key that holds the array
        if isinstance(items, dict):
            for v in items.values():
                if isinstance(v, list):
                    items = v
                    break

        # Extract outermost [...] block and retry (greedy — captures the full tasks array)
        if not isinstance(items, list):
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                # Try json5 on extracted block first
                try:
                    import json5

                    items = json5.loads(match.group())
                    logger.debug("P2-1: JSON5 parse succeeded on extracted [...] block")
                except (ImportError, Exception):
                    # Fallback to standard json
                    items = json.loads(match.group()) if json else None
                except Exception:
                    items = None

        if not isinstance(items, list):
            # Enhanced error logging and recovery
            logger.error(
                "Could not parse decomposition output as JSON. "
                f"Raw response (first 500 chars): {text[:500]!r}"
            )

            # Check if response looks truncated (starts with [ but doesn't end with ])
            text_stripped = text.strip()
            if text_stripped.startswith("[") and not text_stripped.endswith("]"):
                logger.warning("Response appears truncated - attempting partial JSON recovery")
                # Try to extract and complete partial JSON array
                items = self._try_parse_partial_json_array(text_stripped)
                if items and isinstance(items, list) and len(items) > 0:
                    logger.info(f"Recovered {len(items)} tasks from truncated response")
                    # Continue with recovered items - don't return here
                else:
                    logger.error("Partial JSON recovery failed")
                    return {}
            else:
                return {}

        # If we got here with partial recovery, validate items
        if items is None or not isinstance(items, list):
            return {}

        tasks = {}
        for item in items:
            try:
                task_type = TaskType(item["type"])
                prompt = item["prompt"]
                hard_validators = item.get("hard_validators", [])
                target_path = item.get("target_path", "")

                # Strip Python validators for non-Python tasks
                # Detect if task is for HTML/CSS/JS based on prompt keywords or file extension
                (
                    "python" in prompt.lower()
                    or ".py" in target_path.lower()
                    or "flask" in prompt.lower()
                    or "django" in prompt.lower()
                    or "fastapi" in prompt.lower()
                )

                # Priority: If it's explicitly a backend Python task (FastAPI/Flask/Django),
                # keep Python validators even if it mentions HTML/JS (full-stack projects)
                is_backend_python_task = (
                    "fastapi" in prompt.lower()
                    or "flask" in prompt.lower()
                    or "django" in prompt.lower()
                    or "backend" in prompt.lower()
                    and ".py" in target_path.lower()
                )

                is_web_frontend_task = (
                    "html" in prompt.lower()
                    or "css" in prompt.lower()
                    or "javascript" in prompt.lower()
                    or "js" in prompt.lower()
                    or ".html" in target_path.lower()
                    or ".css" in target_path.lower()
                    or ".js" in target_path.lower()
                ) and not is_backend_python_task  # Don't remove if it's a backend task

                # Only remove Python validators for pure frontend tasks, not backend tasks
                if is_web_frontend_task:
                    # Remove Python-specific validators for frontend tasks
                    original_validators = hard_validators
                    hard_validators = [
                        v for v in hard_validators if v not in ("python_syntax", "ruff", "pytest")
                    ]
                    if original_validators != hard_validators:
                        logger.info(
                            f"Task {item['id']}: removed Python validators {set(original_validators) - set(hard_validators)} (web frontend task)"
                        )
                # NOTE: Don't remove validators for non-Python tasks here.
                # Let _filter_validators_for_task() decide based on actual output content.

                task = TaskFactory.create(
                    id=item["id"],
                    task_type=task_type,
                    prompt=prompt,
                    dependencies=item.get("dependencies", []),
                    hard_validators=hard_validators,
                    target_path=target_path,
                    tech_context=item.get("tech_context", ""),
                )
                tasks[task.id] = task
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed task: {e}")
                continue

        logger.info(f"Decomposed into {len(tasks)} tasks")
        return tasks

    # ─────────────────────────────────────────
    # Phase 2-5: Task Execution Loop
    # ─────────────────────────────────────────

    def _check_phase_budget(self, phase: str) -> None:
        """
        Warn when a phase exceeds its soft cap, and log an error when it
        reaches 2× the soft cap (runaway spend in one phase).
        The caps are soft — execution is not halted, but the warnings are
        visible in logs and can be acted upon by the operator.
        """
        spent = self.budget.phase_spent.get(phase, 0.0)
        cap = self.budget.phase_budget(phase)
        if cap <= 0:
            return
        ratio = spent / cap
        if ratio >= 2.0:
            logger.error(
                f"Phase '{phase}' spent ${spent:.4f} — "
                f"{ratio:.1f}× its soft cap of ${cap:.4f}. "
                f"Consider raising --budget or reducing task count."
            )
        elif ratio >= 1.0:
            logger.warning(
                f"Phase '{phase}' exceeded soft cap: " f"${spent:.4f} / ${cap:.4f} ({ratio:.0%})"
            )
            self._hook_registry.fire(
                EventType.BUDGET_WARNING,
                phase=phase,
                spent=spent,
                cap=cap,
                ratio=ratio,
            )

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

        # SkillOpt: fetch best skill doc for injection into system prompt
        skill_prefix = ""
        if self._skill_manager is not None:
            try:
                skill_prefix = await self._skill_manager.best_skill(task.type) or ""
            except Exception:
                pass

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

        # SkillOpt: record trajectory for optimizer (fire-and-forget)
        if self._skill_manager is not None:
            try:
                import asyncio as _asyncio
                import time as _time
                from .models_skill import Trajectory as _Trajectory
                _t = _Trajectory(
                    task_id=task.id,
                    task_type=task.type,
                    prompt=task.prompt[:2000],
                    output=ctx.output[:4000],
                    score=ctx.score,
                    critique_text=ctx.critique[:1000] if ctx.critique else "",
                    model_used=ctx.model.value if ctx.model else "",
                    cost_usd=ctx.cost_usd,
                    recorded_at=_time.time(),
                )
                _task = _asyncio.create_task(self._skill_manager.record_trajectory(_t))
                self._background_tasks.add(_task)
                _task.add_done_callback(self._background_tasks.discard)
            except Exception as _e:
                logger.debug("SkillOpt trajectory skipped: %s", _e)

        return result

    async def _evaluate(self, task: Task, output: str) -> float:
        """Evaluate task quality via EvaluatorService (wired through container)."""
        return await self._evaluator.evaluate(
            task_id=task.id,
            result=output,
            model=task.model if hasattr(task, 'model') else None,
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
        """
        Determine if we should exit early based on stable high performance.

        Exit early if we've seen threshold-level scores with low variance
        across the confidence_window most recent iterations. This saves
        budget on tasks that have already achieved stable good results.

        Args:
            scores_history: List of scores from previous iterations
            threshold: Acceptance threshold for the task
            confidence_window: Number of recent iterations to check (default: 2)
            variance_tolerance: Maximum variance to consider "stable" (default: 0.001)

        Returns:
            True if early exit should occur, False otherwise
        """
        if len(scores_history) < confidence_window:
            return False

        recent = scores_history[-confidence_window:]
        avg_score = sum(recent) / len(recent)

        # Only consider early exit if average is near or above threshold
        if avg_score < threshold * 0.95:
            return False

        # Calculate variance
        variance = sum((s - avg_score) ** 2 for s in recent) / len(recent)

        # Exit if performance is high and stable
        return variance < variance_tolerance

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

        This runs automatically after project completion if _analyze_on_complete=True.
        Results are stored in the Knowledge Base and printed to console.
        """
        try:
            from .project_analyzer import ProjectAnalyzer

            logger.info("🔍 Running post-project analysis...")

            analyzer = ProjectAnalyzer()
            report = await analyzer.analyze_project(
                project_path=output_dir, project_id=state.project_id, run_quality_gate=True
            )

            # Print summary
            summary = analyzer.generate_summary(report)
            logger.info("\n" + summary)

            # Save report to file
            report_file = output_dir / "analysis_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            logger.info(f"📊 Analysis report saved to: {report_file}")

            # Print actionable suggestions
            if report.suggestions:
                print("\n" + "=" * 70)
                print("💡 IMPROVEMENT SUGGESTIONS")
                print("=" * 70)

                for suggestion in report.suggestions[:5]:  # Top 5
                    priority_icon = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🔵",
                    }.get(suggestion.priority.value, "⚪")

                    print(
                        f"\n{priority_icon} [{suggestion.priority.value.upper()}] {suggestion.title}"
                    )
                    print(f"   Category: {suggestion.category.value}")
                    print(f"   Effort: {suggestion.estimated_effort}")
                    print(f"   Impact: {suggestion.expected_impact}")
                    print(f"   {suggestion.description[:100]}...")

                    if suggestion.code_example:
                        print("\n   Example:")
                        for line in suggestion.code_example.strip().split("\n")[:3]:
                            print(f"     {line}")

                print("\n" + "=" * 70)
                print(f"💾 {len(report.suggestions)} suggestions stored in Knowledge Base")
                print("=" * 70)

        except Exception as e:
            logger.warning(f"Project analysis failed: {e}")
            # Don't fail the project if analysis fails

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
