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
import time
from collections import defaultdict, deque
from typing import Optional

from .models import (
    AttemptRecord, Budget, BUDGET_PARTITIONS, FALLBACK_CHAIN, Model, MODEL_MAX_TOKENS, ProjectState,
    ProjectStatus, ROUTING_TABLE, Task, TaskResult, TaskStatus, TaskType,
    get_max_iterations, get_provider, estimate_cost, build_default_profiles,
)
from .api_clients import UnifiedClient, APIResponse
from .validators import run_validators, async_run_validators, all_validators_pass, ValidationResult
from .cache import DiskCache
from .semantic_cache import SemanticCache
try:
    from .cache_optimizer import CacheOptimizer, CacheConfig
    HAS_CACHE_OPTIMIZER = True
except ImportError:
    HAS_CACHE_OPTIMIZER = False
    CacheOptimizer = None
    CacheConfig = None
from .state import StateManager
from .policy import ModelProfile, Policy, PolicySet, JobSpec
from .policy_engine import PolicyEngine

# Test validation for reliable test generation
try:
    from .test_validator import TestValidator, validate_and_generate_test
    HAS_TEST_VALIDATOR = True
except ImportError as _e:
    HAS_TEST_VALIDATOR = False
    TestValidator = None
    validate_and_generate_test = None

# Code validation for clean code generation (no LLM commentary)
try:
    from .code_validator import validate_code, extract_code_from_llm_response
    HAS_CODE_VALIDATOR = True
except ImportError as _e:
    HAS_CODE_VALIDATOR = False
    validate_code = None
    extract_code_from_llm_response = None
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector
from .audit import AuditLog
from .optimization import OptimizationBackend
from .hooks import HookRegistry, EventType
from .metrics import MetricsExporter
from .agents import TaskChannel
from .cost import BudgetHierarchy, CostPredictor
from .tracing import traced_task, get_tracer, TracingConfig, configure_tracing
from .telemetry_store import TelemetryStore
# NEW: Security & Accountability modules from "Agents of Chaos" paper (arXiv:2602.20021)
from .task_verifier import TaskVerifier
from .accountability import AccountabilityTracker, ActorType, ActionType
from .agent_safety import AgentSafetyMonitor, SafetyEventType
from .red_team import RedTeamFramework
# NEW: External Projects Integration (RTK, Mnemo Cortex, LiteLLM)
from .token_optimizer import TokenOptimizer, get_optimizer
from .preflight import PreflightValidator, PreflightMode, get_validator
from .session_watcher import SessionWatcher, get_session_watcher
from .persona import PersonaManager, PersonaMode, get_persona_manager
from .memory_tier import MemoryTierManager, get_memory_manager
from .bm25_search import BM25Search, get_bm25_search
from .reranker import LLMReranker, get_reranker
from .a2a_protocol import A2AManager, AgentCard, get_a2a_manager

logger = logging.getLogger("orchestrator")


def _clean_code_output(text: str, task_type: TaskType) -> str:
    """
    Post-process code output to remove common LLM artifacts:
    - Markdown fences (```language...```)
    - Placeholder comments explaining what to add
    - Explanatory text fragments that aren't code
    """
    if task_type != TaskType.CODE_GEN:
        return text
    
    # Remove markdown code fences
    text = re.sub(r'^```\w*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common placeholder/explanatory comment patterns
    # Matches: // Add content that can be easily replaced...
    #          /* Add your code here */
    #          <!-- Placeholder for ... -->
    placeholder_patterns = [
        r'//\s*[Aa]dd\s+(?:content|code|your|more|placeholder).*?\n',
        r'//\s*[Rr]eplace\s+this.*?(?:\n|$)',
        r'//\s*[Tt]ODO:.*?(?:\n|$)',
        r'//\s*[Ff]IXME:.*?(?:\n|$)',
        r'/\*\s*[Aa]dd\s+(?:content|code|your).*?\*/',
        r'/\*\s*[Rr]eplace\s+this.*?\*/',
        r'<!--\s*[Aa]dd\s+(?:content|code|your).*?-->',
        r'<!--\s*[Rr]eplace\s+this.*?-->',
        r'#\s*[Aa]dd\s+(?:content|code|your|more).*?(?:\n|$)',
        r'#\s*[Rr]eplace\s+this.*?(?:\n|$)',
    ]
    
    for pattern in placeholder_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up multiple consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


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

    def __init__(self, budget: Optional[Budget] = None,
                 cache: Optional[DiskCache] = None,
                 state_manager: Optional[StateManager] = None,
                 max_concurrency: int = 3,
                 max_parallel_tasks: int = 3,
                 budget_hierarchy: Optional["BudgetHierarchy"] = None,
                 cost_predictor: Optional["CostPredictor"] = None,
                 tracing_cfg: Optional["TracingConfig"] = None,
                 telemetry_store: Optional["TelemetryStore"] = None):
        self.budget = budget or Budget()
        self.cache = cache or DiskCache()
        self.state_mgr = state_manager or StateManager()
        self.client = UnifiedClient(cache=self.cache, max_concurrency=max_concurrency)
        
        # NEW: Multi-level cache optimizer (L1/L2/L3)
        if HAS_CACHE_OPTIMIZER:
            self._cache_optimizer = CacheOptimizer(CacheConfig(
                l1_max_size=200,
                l1_ttl_seconds=3600,
                l2_ttl_hours=48,
                l3_quality_threshold=0.85,
                track_stats=True,
            ))
        else:
            self._cache_optimizer = None
        self.api_health: dict[Model, bool] = {m: True for m in Model}
        self.results: dict[str, TaskResult] = {}
        self._results_lock = asyncio.Lock()  # Protects concurrent access to self.results
        self._project_id: str = ""
        # Max tasks executed concurrently within one dependency level.
        # JobSpec.max_parallel_tasks overrides this via run_job().
        self._max_parallel_tasks: int = max(1, max_parallel_tasks)

        # Post-project analysis flag
        self._analyze_on_complete: bool = False

        # Circuit breaker counters — consecutive failures per model
        self._consecutive_failures: dict[Model, int] = {m: 0 for m in Model}

        # BUG-SHUTDOWN-001 FIX: Track fire-and-forget background tasks for proper shutdown
        self._background_tasks: set[asyncio.Task] = set()

        for model in Model:
            if not self.client.is_available(model):
                self.api_health[model] = False
                logger.warning(f"{model.value}: provider SDK/key not available")

        # Policy-driven components (initialised with default profiles from static tables)
        self._profiles: dict[Model, ModelProfile] = build_default_profiles()
        self._audit_log = AuditLog()
        self._policy_engine = PolicyEngine(audit_log=self._audit_log)
        self._planner = ConstraintPlanner(
            profiles=self._profiles,
            policy_engine=self._policy_engine,
            api_health=self.api_health,
        )
        self._telemetry = TelemetryCollector(self._profiles)
        # Active policy set — replaced by run_job(); empty = no restrictions
        self._active_policies: PolicySet = PolicySet()
        # Context truncation limit per dependency (chars) — configurable.
        # Raised from 20000: code_generation outputs routinely reach 25000+ chars
        # and truncation causes code_review tasks to miss the tail of the source,
        # leading the LLM to claim "source code was not provided".
        self.context_truncation_limit: int = 40000
        # Improvement 3: event hooks + metrics exporter
        self._hook_registry: HookRegistry = HookRegistry()
        self._metrics_exporter: Optional[MetricsExporter] = None
        # Improvement 5: named TaskChannels for inter-task messaging
        self._channels: dict[str, TaskChannel] = {}
        # Improvement 6: cross-run budget hierarchy + adaptive cost predictor
        self._budget_hierarchy: Optional[BudgetHierarchy] = budget_hierarchy
        self._cost_predictor: Optional[CostPredictor] = cost_predictor
        # Task 2: streaming event bus (None unless run_project_streaming() is active)
        self._event_bus: Optional["ProjectEventBus"] = None
        # Task 6: adaptive router v2 — circuit breaker with degraded/disabled states
        from .adaptive_router import AdaptiveRouter
        self._adaptive_router = AdaptiveRouter()
        # Task 7: configure OpenTelemetry tracing if a config was provided
        if tracing_cfg is not None:
            configure_tracing(tracing_cfg)
        # Persistent cross-run learning store (Learn & Show feature)
        self._telemetry_store: TelemetryStore = (
            telemetry_store if telemetry_store is not None else TelemetryStore()
        )
        # OPTIMIZATION: Semantic cache for high-level pattern reuse
        # Note: Now integrated into CacheOptimizer (L3 cache)
        self._semantic_cache = SemanticCache(quality_threshold=0.85)
        # Track if we've been entered as a context manager
        self._entered: bool = False
        
        # Dashboard integration (v5.1)
        self._dashboard_integration: Optional[Any] = None
        self._architecture_rules: Optional[Any] = None
        
        # Git integration (auto-commit after tasks)
        self._git_integration: Optional[Any] = None
        self._output_dir: Optional[Path] = None

        # NEW: Security & Accountability modules (arXiv:2602.20021)
        # Task Verification - prevents task completion misrepresentation
        self._task_verifier: TaskVerifier = TaskVerifier()
        # Accountability - tracks action attribution and downstream impacts
        self._accountability: AccountabilityTracker = AccountabilityTracker()
        # Agent Safety - prevents cross-agent unsafe practice propagation
        self._agent_safety: AgentSafetyMonitor = AgentSafetyMonitor()
        # Red-Team Framework - stress testing methodology
        self._red_team: RedTeamFramework = RedTeamFramework()
        
        # NEW: External Projects Integration (RTK, Mnemo Cortex, LiteLLM)
        # Token Optimizer - CLI output filtering (60-90% token savings)
        self._token_optimizer: TokenOptimizer = TokenOptimizer()
        # Preflight Validator - response quality control (PASS/ENRICH/WARN/BLOCK)
        self._preflight_validator: PreflightValidator = PreflightValidator()
        # Session Watcher - auto-capture conversations
        self._session_watcher: SessionWatcher = SessionWatcher()
        # Persona Manager - behavior customization
        self._persona_manager: PersonaManager = PersonaManager()
        # Memory Tier Manager - HOT/WARM/COLD memory hierarchy with BM25
        self._memory_manager: MemoryTierManager = MemoryTierManager(enable_bm25=True)
        # BM25 Search - SQLite FTS5 full-text search
        self._bm25_search: BM25Search = get_bm25_search(str(self._memory_manager.storage_path / "search.db"))
        # LLM Re-ranker - quality-based result re-ranking
        self._reranker: LLMReranker = get_reranker()
        # A2A Manager - agent-to-agent communication
        self._a2a_manager: A2AManager = A2AManager()

    # ─────────────────────────────────────────
    # Async Context Manager
    # ─────────────────────────────────────────

    async def __aenter__(self) -> "Orchestrator":
        """
        Enter async context manager.
        
        Ensures all resources are properly initialized and will be cleaned up
        on exit. Use this pattern for guaranteed resource cleanup:
        
            async with Orchestrator() as orch:
                result = await orch.run_project(...)
        
        Returns:
            Self for use in async with statement
        """
        self._entered = True
        logger.debug("Orchestrator entered as context manager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager, ensuring all resources are cleaned up.

        Cleanup order:
        1. Clean up completed background tasks (BUG-MEMORY-002 FIX)
        2. Wait for pending background tasks (BUG-SHUTDOWN-001 FIX)
        3. Flush any pending telemetry snapshots
        4. Close cache connection
        5. Close state manager connection
        6. Flush audit log
        7. Flush telemetry store

        Exceptions during cleanup are logged but not raised to avoid masking
        the original exception.

        BUG-EVENTLOOP-001 FIX: Properly wait for aiosqlite background threads
        to complete before event loop closes.
        """
        logger.debug("Orchestrator exiting context manager, cleaning up resources...")

        # BUG-MEMORY-002 FIX: Clean up completed background tasks first
        await self._cleanup_background_tasks()

        # BUG-SHUTDOWN-001 FIX: Wait for background tasks to complete
        if self._background_tasks:
            logger.debug(f"Waiting for {len(self._background_tasks)} background tasks...")
            if self._background_tasks:
                done, pending = await asyncio.wait(
                    self._background_tasks,
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
            if hasattr(self._audit_log, 'flush'):
                await self._audit_log.flush()
                logger.debug("Audit log flushed")
        except Exception as e:
            logger.warning(f"Failed to flush audit log: {e}")

        # 5. Flush telemetry store
        try:
            await self._telemetry_store.flush()
            logger.debug("Telemetry store flushed")
        except Exception as e:
            logger.warning(f"Failed to flush telemetry store: {e}")

        self._entered = False
        logger.debug("Orchestrator cleanup complete")

    async def close(self) -> None:
        """
        Explicitly close all resources.
        
        Called automatically when using async context manager (async with),
        but can also be called explicitly for manual resource management.
        """
        await self.__aexit__(None, None, None)

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
                profile.quality_score  = hist.quality_score
                profile.trust_factor   = hist.trust_factor
                profile.avg_latency_ms = hist.avg_latency_ms
                profile.latency_p95_ms = hist.latency_p95_ms
            else:
                # WARM: 40% historical / 60% default blend
                profile.quality_score = 0.4 * hist.quality_score + 0.6 * profile.quality_score
                profile.trust_factor  = 0.4 * hist.trust_factor  + 0.6 * profile.trust_factor

    async def _flush_telemetry_snapshots(self, project_id: str) -> None:
        """
        Fire-and-forget: snapshot each ModelProfile that was used this run.
        Only profiles with call_count >= 1 are written.
        Uses asyncio.create_task so the hot path is never blocked.

        BUG-SHUTDOWN-001 FIX: Task is tracked for proper shutdown waiting.
        BUG-MEMORY-002 FIX: Added exception handling in callback to prevent leaks.
        """
        from .models import TaskType as _TT

        async def _write_snapshots() -> None:
            for model, profile in self._profiles.items():
                if profile.call_count < 1:
                    continue
                try:
                    await self._telemetry_store.record_snapshot(
                        project_id, model, _TT.CODE_GEN, profile
                    )
                except Exception as exc:
                    logger.warning("TelemetryStore.record_snapshot failed: %s", exc)

        # BUG-MEMORY-002 FIX: Wrap callback with exception handling
        def _cleanup_task(task: asyncio.Task) -> None:
            """Safely remove task from tracking set."""
            try:
                self._background_tasks.discard(task)
            except Exception as e:
                # Log but don't propagate - set may be modified during iteration
                logger.warning(f"Failed to remove background task from tracking: {e}")
            
            # Log completion for debugging
            if task.cancelled():
                logger.debug("Background task was cancelled")
            elif task.exception() is not None:
                logger.warning(f"Background task completed with exception: {task.exception()}")
            else:
                logger.debug("Background task completed successfully")

        # BUG-SHUTDOWN-001 FIX: Track background task
        task = asyncio.create_task(_write_snapshots())
        self._background_tasks.add(task)
        task.add_done_callback(_cleanup_task)

    async def _cleanup_background_tasks(self) -> int:
        """
        BUG-MEMORY-002 FIX: Periodic cleanup of completed background tasks.
        
        This method removes completed tasks from the tracking set to prevent
        memory leaks. It should be called periodically or during shutdown.
        
        Returns:
            Number of tasks cleaned up
        """
        if not self._background_tasks:
            return 0
        
        # Find completed tasks
        completed = {task for task in self._background_tasks if task.done()}
        
        # Remove completed tasks
        for task in completed:
            # Ensure callback ran (it should have removed the task)
            # But double-check in case callback failed
            self._background_tasks.discard(task)
            
            # Log if task had exception and callback didn't catch it
            if task.exception() is not None:
                logger.warning(
                    f"Cleaned up background task with unhandled exception: {task.exception()}"
                )
        
        if completed:
            logger.debug(f"Cleaned up {len(completed)} completed background tasks")
        
        return len(completed)

    async def _safe_record_routing_event(
        self,
        project_id: str,
        task_id: str,
        task_type: "TaskType",
        result: "TaskResult",
    ) -> None:
        """Fire-and-forget wrapper: record a routing event, swallowing exceptions."""
        try:
            await self._telemetry_store.record_routing_event(
                project_id, task_id, task_type, result
            )
        except Exception as exc:
            logger.warning("TelemetryStore.record_routing_event failed: %s", exc)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def set_optimization_backend(self, backend: "OptimizationBackend") -> None:
        """Swap the ConstraintPlanner's optimization strategy at runtime."""
        self._planner.set_backend(backend)

    @property
    def audit_log(self) -> "AuditLog":
        """Read-only access to the policy audit log."""
        return self._audit_log

    @property
    def cost_predictor(self) -> Optional["CostPredictor"]:
        """Read-only access to the CostPredictor, if one was configured."""
        return self._cost_predictor

    def add_hook(self, event: str, callback) -> None:
        """Register an event hook callback. See orchestrator.hooks.EventType for event names."""
        self._hook_registry.add(event, callback)

    def set_metrics_exporter(self, exporter: "MetricsExporter") -> None:
        """Set the MetricsExporter to use when export_metrics() is called."""
        self._metrics_exporter = exporter

    def export_metrics(self) -> None:
        """Export live per-model telemetry stats via the configured MetricsExporter."""
        if self._metrics_exporter is None:
            return
        self._metrics_exporter.export(self._build_metrics_dict())

    def get_channel(self, name: str) -> "TaskChannel":
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
        context: Optional[dict] = None,
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
        query: Optional[str] = None,
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
            return [r.to_dict() if hasattr(r, 'to_dict') else r for r in reranked]
        
        return memories[:limit]

    async def hybrid_search(
        self,
        query: str,
        project_id: str,
        limit: int = 10,
        use_reranking: bool = True,
    ) -> list:
        """
        Perform hybrid search with BM25 + optional re-ranking.
        
        Args:
            query: Search query
            project_id: Project to search
            limit: Maximum results
            use_reranking: Apply LLM re-ranking
            
        Returns:
            Search results ordered by relevance
        """
        # BM25 search
        bm25_results = await self._bm25_search.bm25_search(
            query=query,
            project_id=project_id,
            limit=limit * 2 if use_reranking else limit,
        )
        
        results = [r.to_dict() for r in bm25_results]
        
        # Apply re-ranking
        if use_reranking and results:
            reranked = await self._reranker.rerank(query, results, top_k=limit)
            return [r.to_dict() for r in reranked]
        
        return results[:limit]

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
        context: Optional[dict] = None,
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
        expected_outputs: Optional[list[str]] = None,
        required_patterns: Optional[list[str]] = None,
        forbidden_patterns: Optional[list[str]] = None,
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
        """Notify dashboard of project start."""
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_project_start(
                    project_id, state, self._architecture_rules
                )
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")
    
    def _notify_dashboard_task_start(self, task_id: str, task: Task, model: Optional[Model]):
        """Notify dashboard of task start."""
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_task_start(task_id, task, model)
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")
    
    def _notify_dashboard_task_progress(self, iteration: int, score: float):
        """Notify dashboard of task progress."""
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_task_progress(iteration, score)
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")
    
    def _notify_dashboard_task_complete(self, task_id: str, status: str):
        """Notify dashboard of task completion."""
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_task_complete(task_id, status)
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")

    def _build_metrics_dict(self) -> dict:
        """Build a per-model metrics dict from live ModelProfile data."""
        result: dict = {}
        for model, profile in self._profiles.items():
            result[model.value] = {
                "call_count":           profile.call_count,
                "failure_count":        profile.failure_count,
                "success_rate":         profile.success_rate,
                "avg_latency_ms":       profile.avg_latency_ms,
                "latency_p95_ms":       profile.latency_p95_ms,
                "quality_score":        profile.quality_score,
                "trust_factor":         profile.trust_factor,
                "avg_cost_usd":         profile.avg_cost_usd,
                "validator_fail_count": profile.validator_fail_count,
                "error_rate":           self._telemetry.error_rate(model),
            }
        return result

    async def run_project(self, project_description: str,
                          success_criteria: str,
                          project_id: str = "",
                          app_profile: Optional["AppProfile"] = None,
                          analyze_on_complete: bool = False,
                          output_dir: Optional[Path] = None) -> ProjectState:
        """
        Main entry point. Decomposes project → executes tasks → returns state.
        
        Args:
            project_description: What to build
            success_criteria: How to verify success
            project_id: Optional project identifier
            app_profile: Optional application profile
            analyze_on_complete: If True, run post-project analysis
            output_dir: Directory containing project output (for analysis)
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("run_project") as span:
            span.set_attribute("project.description", project_description[:200])
            if not project_id:
                project_id = hashlib.md5(
                    f"{project_description[:100]}{time.time()}".encode()
                ).hexdigest()[:12]
            self._project_id = project_id

            logger.info(f"Starting project {project_id}")
            logger.info(f"Budget: ${self.budget.max_usd}, {self.budget.max_time_seconds}s")

            try:
                # Check if resumable
                existing = await self.state_mgr.load_project(project_id)
                if existing and existing.status == ProjectStatus.PARTIAL_SUCCESS:
                    logger.info(f"Resuming project {project_id} from checkpoint")
                    state = await self._resume_project(existing)
                    await self.state_mgr.save_project(project_id, state)
                    self._log_summary(state)
                    return state

                # Phase 0: Architecture Decision & Rules Generation
                architecture_rules = await self._generate_architecture_rules(
                    project_description, success_criteria, output_dir
                )
                self._architecture_rules = architecture_rules

                # Phase 1: Decompose
                tasks = await self._decompose(project_description, success_criteria,
                                              app_profile=app_profile)
                if not tasks:
                    return self._make_state(
                        project_description, success_criteria, {},
                        ProjectStatus.SYSTEM_FAILURE
                    )

                # Topological sort
                execution_order = self._topological_sort(tasks)
                logger.info(f"Execution order: {execution_order}")

                # Create initial state for dashboard
                initial_state = self._make_state(
                    project_description, success_criteria, tasks,
                    execution_order=execution_order
                )
                
                # Notify dashboard of project start
                logger.debug("Notifying dashboard of project start...")
                self._notify_dashboard_project_start(project_id, initial_state)
                logger.debug("Dashboard notification complete")

                # Emit ProjectStarted streaming event
                if self._event_bus:
                    from .events import ProjectStartedEvent
                    logger.debug("Publishing ProjectStarted event...")
                    await self._event_bus.publish(ProjectStartedEvent(
                        project_id=self._project_id,
                        description=project_description[:200],
                        budget=self.budget.max_usd,
                        budget_usd=self.budget.max_usd,
                        total_tasks=len(tasks),
                    ))
                    logger.debug("ProjectStarted event published")

                # Phase 2-5: Execute
                logger.info("Starting task execution...")
                state = await self._execute_all(
                    tasks, execution_order, project_description, success_criteria
                )
                logger.info("Task execution completed.")

                # Final status determination
                state.execution_order = execution_order
                state.status = self._determine_final_status(state)
                await self.state_mgr.save_project(project_id, state)

                self._log_summary(state)
                
                # Final Git commit for project completion
                if (self._git_integration is not None and 
                    self._git_integration.is_available()):
                    try:
                        total_tasks = len(tasks)
                        completed = sum(1 for r in self.results.values() 
                                      if r.status == TaskStatus.COMPLETED)
                        commit_hash = self._git_integration.commit_project(
                            project_name=project_description[:50],
                            total_tasks=total_tasks,
                            total_cost=self.budget.spent_usd,
                            elapsed_time=self.budget.elapsed_seconds,
                        )
                        if commit_hash:
                            logger.info(f"Final git commit: {commit_hash}")
                            branch = self._git_integration.get_branch_name()
                            logger.info(f"Branch: {branch}")
                    except Exception as e:
                        logger.warning(f"Final git commit failed: {e}")

                # Emit ProjectCompleted streaming event
                if self._event_bus:
                    from .events import ProjectCompletedEvent
                    completed_count = sum(1 for r in self.results.values()
                                          if r.status != TaskStatus.FAILED)
                    failed_count = sum(1 for r in self.results.values()
                                       if r.status == TaskStatus.FAILED)
                    await self._event_bus.publish(ProjectCompletedEvent(
                        project_id=self._project_id,
                        status=state.status.value,
                        total_cost=self.budget.spent_usd,
                        total_cost_usd=self.budget.spent_usd,
                        duration_seconds=self.budget.elapsed_seconds,
                        elapsed_seconds=self.budget.elapsed_seconds,
                        tasks_completed=completed_count,
                        tasks_failed=failed_count,
                    ))

                # Post-project analysis and improvement suggestions
                if analyze_on_complete and output_dir:
                    await self._analyze_completed_project(state, output_dir)

                return state

            finally:
                # Always close both DB connections so aiosqlite background threads
                # finish their callbacks before asyncio.run() closes the loop.
                await self.state_mgr.close()
                await self.cache.close()

    async def run_job(self, spec: JobSpec) -> ProjectState:
        """
        Policy-driven entry point. Accepts a JobSpec that bundles project
        description, success criteria, budget, quality targets, and policies.

        The active PolicySet is threaded through model selection so that
        ConstraintPlanner enforces compliance on every API call.
        """
        self.budget = spec.budget
        self._active_policies = spec.policy_set
        # JobSpec may override the per-task parallelism limit
        if spec.max_parallel_tasks > 0:
            self._max_parallel_tasks = spec.max_parallel_tasks
        # Warm-start: blend historical profiles before execution
        await self._apply_warm_start()
        # BudgetHierarchy pre-flight check (Improvement 6)
        if self._budget_hierarchy is not None:
            job_id = getattr(spec, "job_id", "") or ""
            team   = getattr(spec, "team",   "") or ""
            if not self._budget_hierarchy.can_afford_job(job_id, team, spec.budget.max_usd):
                raise ValueError(
                    f"BudgetHierarchy rejects job '{job_id}': "
                    "org/team/job limits would be exceeded"
                )
        state = await self.run_project(
            project_description=spec.project_description,
            success_criteria=spec.success_criteria,
        )
        # Charge actual spend to BudgetHierarchy so cross-run caps are enforced.
        if self._budget_hierarchy is not None:
            actual_spend = self.budget.max_usd - self.budget.remaining_usd
            job_id = getattr(spec, "job_id", "") or ""
            team   = getattr(spec, "team",   "") or ""
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

    async def dry_run(self, project_description: str,
                      success_criteria: str) -> "ExecutionPlan":
        """
        Dry-run: decompose the project, build an execution plan, and return it
        WITHOUT executing any tasks. (Improvement 12)

        Makes one real API call (decomposition) then stops. No task execution
        or state persistence happens.

        Returns an ExecutionPlan that can be printed with plan.render().
        """
        from .dry_run import (
            ExecutionPlan, TaskPlan, _TOKEN_ESTIMATES, _DEFAULT_TOKENS,
        )
        from .models import estimate_cost, ROUTING_TABLE

        tasks = await self._decompose(project_description, success_criteria)
        if not tasks:
            return ExecutionPlan(
                project_description=project_description,
                success_criteria=success_criteria,
            )

        levels = self._topological_levels(tasks)
        # Build a level_index map: task_id → level
        level_index: dict[str, int] = {}
        for lvl_idx, lvl_tasks in enumerate(levels):
            for tid in lvl_tasks:
                level_index[tid] = lvl_idx

        task_plans: list[TaskPlan] = []
        total_cost = 0.0

        for tid, task in tasks.items():
            model_list = ROUTING_TABLE.get(task.type, [])
            available = [m for m in model_list if self.api_health.get(m, True)]
            primary = available[0] if available else (model_list[0] if model_list else None)

            in_tokens, out_tokens = _TOKEN_ESTIMATES.get(
                task.type.value, _DEFAULT_TOKENS
            )
            cost = estimate_cost(primary, in_tokens, out_tokens) if primary else 0.0
            total_cost += cost

            task_plans.append(TaskPlan(
                task_id=tid,
                task_type=task.type.value,
                prompt_preview=(task.prompt[:80].replace("\n", " ") + "…"
                                if len(task.prompt) > 80 else task.prompt),
                dependencies=list(task.dependencies),
                parallel_level=level_index.get(tid, 0),
                primary_model=primary.value if primary else "unknown",
                estimated_cost_usd=round(cost, 6),
                acceptance_threshold=task.acceptance_threshold,
                max_iterations=task.max_iterations,
            ))

        # Sort by (level, task_id) so render is deterministic
        task_plans.sort(key=lambda t: (t.parallel_level, t.task_id))

        return ExecutionPlan(
            project_description=project_description,
            success_criteria=success_criteria,
            tasks=task_plans,
            parallel_levels=levels,
            estimated_total_cost=round(total_cost, 6),
            num_parallel_levels=len(levels),
        )

    # ─────────────────────────────────────────
    # Phase 1: Decomposition
    # ─────────────────────────────────────────

    async def _decompose(self, project: str, criteria: str,
                          app_profile: Optional["AppProfile"] = None) -> dict[str, Task]:
        """Use cheapest capable model to break project into atomic tasks."""
        valid_types = [t.value for t in TaskType]

        # Build optional app-context block injected into the prompt
        app_context_block = ""
        if app_profile is not None:
            from orchestrator.scaffold import _TEMPLATE_MAP
            from orchestrator.scaffold.templates import generic
            template_files = _TEMPLATE_MAP.get(app_profile.app_type, generic.FILES)
            scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
            tech_stack_str = ", ".join(app_profile.tech_stack) if app_profile.tech_stack else "unknown"

            # Build architecture block if ArchitectureDecision fields are present
            arch_block = ""
            if hasattr(app_profile, "structural_pattern") and app_profile.structural_pattern:
                rationale_line = (
                    f"\n  Rationale:          {app_profile.rationale}"
                    if getattr(app_profile, "rationale", "") else ""
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

        prompt = f"""You are a project decomposition engine. Break this project into
atomic, executable tasks.

PROJECT: {project}

SUCCESS CRITERIA: {criteria}
{app_context_block}
Return ONLY a JSON array. Each element must have:
- "id": string (e.g., "task_001")
- "type": one of {valid_types}
- "prompt": detailed instruction for the task executor. For code_generation tasks, MUST include:\n"
  "  - Code must be THOROUGHLY COMMENTED\n"
  "  - EVERY file MUST start with: /** Author: Georgios-Chrysovalantis Chatzivantsidis */\n"
- "dependencies": list of task id strings this depends on (empty if none)
- "hard_validators": list of validator names — ONLY use these for code tasks:
  - "python_syntax": only for code_generation tasks that produce Python code
  - "json_schema": only for tasks that must return valid JSON
  - "pytest": only for code_generation tasks with runnable tests
  - "ruff": only for code_generation tasks requiring lint checks
  - "latex": only for tasks producing LaTeX documents
  - "length": for tasks requiring minimum/maximum output length
  - Use [] (empty list) for non-code tasks (reasoning, writing, analysis, evaluation)

RULES:
- Tasks must be atomic (one clear deliverable each)
- Dependencies must form a DAG (no cycles)
- Include code_review tasks after code_generation tasks
- Include at least one evaluation task at the end
- 5-15 tasks total for a medium project
- Do NOT add hard_validators to reasoning, writing, analysis, or evaluation tasks

Return ONLY the JSON array, no markdown fences, no explanation."""

        decomp_system = "You are a precise project decomposition engine. Output only valid JSON."
        # Use fast, reliable models for decomposition (not just cheapest)
        # Priority: GPT-4o-mini (fast, reliable), Gemini Flash (fast), then cheapest
        model = self._get_fast_decomposition_model()

        async def _try_decompose(m: Model) -> dict[str, Task]:
            resp = await self.client.call(
                m, prompt, system=decomp_system, max_tokens=4096, timeout=45,
                bypass_cache=True,  # never reuse a cached decomposition response
            )
            self.budget.charge(resp.cost_usd, "decomposition")
            self._record_success(m, resp)
            result = self._parse_decomposition(resp.text)
            if not result:
                raise ValueError(f"Decomposition returned empty task list from {m.value}")
            return result

        # Try primary model, then fallback, with one retry on empty/malformed output
        for attempt, m in enumerate([model, self._get_fallback(model)]):
            if m is None:
                break
            try:
                return await _try_decompose(m)
            except (Exception, asyncio.CancelledError) as e:
                logger.error(f"Decomposition attempt {attempt + 1} with {m.value} failed: {e}")
                self._record_failure(m, error=e)
        logger.error("All decomposition attempts failed — returning empty task list")
        return {}

    def _parse_decomposition(self, text: str) -> dict[str, Task]:
        """Parse LLM output into Task objects with defensive handling."""
        text = text.strip()

        # Strip markdown fences (``` or ```json)
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        def _try_parse(s: str):
            """Attempt json.loads; on failure try progressively more aggressive fixes."""
            # 1. Direct parse
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # 2. Strip trailing commas before ] or } (common LLM mistake)
            cleaned = re.sub(r',\s*([}\]])', r'\1', s)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            # 3. Remove control characters (except \n \r \t)
            cleaned2 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
            try:
                return json.loads(cleaned2)
            except json.JSONDecodeError:
                pass
            return None

        # Try full text first
        items = _try_parse(text)

        # If the top-level is a dict, look for a key that holds the array
        if isinstance(items, dict):
            for v in items.values():
                if isinstance(v, list):
                    items = v
                    break

        # Extract outermost [...] block and retry (greedy — captures the full tasks array)
        if not isinstance(items, list):
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                items = _try_parse(match.group())

        if not isinstance(items, list):
            logger.error(
                "Could not parse decomposition output as JSON. "
                f"Raw response (first 500 chars): {text[:500]!r}"
            )
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
                is_python_task = (
                    "python" in prompt.lower() or
                    ".py" in target_path.lower() or
                    "flask" in prompt.lower() or
                    "django" in prompt.lower() or
                    "fastapi" in prompt.lower()
                )
                
                # Priority: If it's explicitly a backend Python task (FastAPI/Flask/Django),
                # keep Python validators even if it mentions HTML/JS (full-stack projects)
                is_backend_python_task = (
                    "fastapi" in prompt.lower() or
                    "flask" in prompt.lower() or
                    "django" in prompt.lower() or
                    "backend" in prompt.lower() and ".py" in target_path.lower()
                )
                
                is_web_frontend_task = (
                    ("html" in prompt.lower() or
                    "css" in prompt.lower() or
                    "javascript" in prompt.lower() or
                    "js" in prompt.lower() or
                    ".html" in target_path.lower() or
                    ".css" in target_path.lower() or
                    ".js" in target_path.lower())
                    and not is_backend_python_task  # Don't remove if it's a backend task
                )
                
                # Only remove Python validators for pure frontend tasks, not backend tasks
                if is_web_frontend_task:
                    # Remove Python-specific validators for frontend tasks
                    original_validators = hard_validators
                    hard_validators = [v for v in hard_validators if v not in ("python_syntax", "ruff", "pytest")]
                    if original_validators != hard_validators:
                        logger.info(f"Task {item['id']}: removed Python validators {set(original_validators) - set(hard_validators)} (web frontend task)")
                # NOTE: Don't remove validators for non-Python tasks here.
                # Let _filter_validators_for_task() decide based on actual output content.
                
                task = Task(
                    id=item["id"],
                    type=task_type,
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
                f"Phase '{phase}' exceeded soft cap: "
                f"${spent:.4f} / ${cap:.4f} ({ratio:.0%})"
            )
            self._hook_registry.fire(
                EventType.BUDGET_WARNING,
                phase=phase, spent=spent, cap=cap, ratio=ratio,
            )

    async def _execute_all(self, tasks: dict[str, Task],
                            execution_order: list[str],
                            project_desc: str,
                            success_criteria: str,
                            output_dir: "Optional[Path]" = None) -> ProjectState:
        """
        Execute all tasks respecting dependencies, with intra-level parallelism.

        Tasks are grouped into dependency levels using _topological_levels().
        All tasks in the same level have no inter-dependencies and are executed
        concurrently up to self._max_parallel_tasks simultaneous coroutines.

        A semaphore limits concurrency so that API rate limits and memory usage
        remain manageable even when many tasks are eligible at once.
        """
        logger.info("Building task execution levels...")
        levels = self._topological_levels(tasks)
        logger.info(f"Built {len(levels)} execution levels")
        semaphore = asyncio.Semaphore(self._max_parallel_tasks)

        # Improvement 13: progressive output writer (writes after each task)
        _progress_writer = None
        _prog_output = None
        if output_dir is not None:
            from .progress_writer import ProgressWriter
            from .progressive_output import ProgressiveOutputManager
            from .git_integration import GitIntegration, get_default_git_config
            
            logger.info(f"Initializing progress writer for output: {output_dir}")
            # Build a temporary ProjectState reference for summary.json writes
            _partial_state = self._make_state(
                project_desc, success_criteria, tasks,
                execution_order=execution_order,
            )
            # Share the live results dict so summary.json always reflects current state
            _partial_state.results = self.results
            _progress_writer = ProgressWriter(Path(output_dir), _partial_state)
            
            # Initialize progressive output manager (bmalph-style)
            _prog_output = ProgressiveOutputManager(
                Path(output_dir), 
                project_name=project_desc[:30].replace(" ", "_")
            )
            logger.info(f"Progressive output manager initialized in {output_dir}/outputs/")
            
            # Initialize Git integration for auto-commits
            git_config = get_default_git_config()
            self._git_integration = GitIntegration(output_dir, git_config)
            if self._git_integration.is_available():
                branch = self._git_integration.setup_project_branch(
                    self._project_id, project_desc[:40]
                )
                if branch:
                    logger.info(f"Git integration active on branch: {branch}")
            
            logger.info("Progress writer initialized")

        async def _run_one(task_id: str) -> None:
            """Execute a single task under the concurrency semaphore."""
            logger.debug(f"Acquiring semaphore for task {task_id}...")
            async with semaphore:
                logger.debug(f"Executing task {task_id}")
                
                # Check budget with lock protection
                async with self._results_lock:
                    budget_ok = self.budget.can_afford(0.01)
                    time_ok = self.budget.time_remaining()
                
                if not budget_ok:
                    logger.warning(f"Budget exhausted, skipping {task_id}")
                    async with self._results_lock:
                        self.results[task_id] = TaskResult(
                            task_id=task_id, output="", score=0.0,
                            model_used=Model.GPT_4O_MINI,
                            status=TaskStatus.FAILED,
                        )
                    return
                if not time_ok:
                    logger.warning(f"Time limit reached, skipping {task_id}")
                    async with self._results_lock:
                        self.results[task_id] = TaskResult(
                            task_id=task_id, output="", score=0.0,
                            model_used=Model.GPT_4O_MINI,
                            status=TaskStatus.FAILED,
                        )
                    return

                task = tasks[task_id]
                self._hook_registry.fire(EventType.TASK_STARTED, task_id=task_id, task=task)

                # Get the primary model for this task type for dashboard
                task_models = self._get_available_models(task.type)
                primary_model = task_models[0] if task_models else None
                self._notify_dashboard_task_start(task_id, task, primary_model)

                result = await self._execute_task(task)
                
                # BUG-RACE-002 FIX: Protect results dict with lock
                async with self._results_lock:
                    self.results[task_id] = result
                task.status = result.status

                # Notify dashboard of task completion
                self._notify_dashboard_task_complete(task_id, result.status.value)

                self._hook_registry.fire(EventType.TASK_COMPLETED, task_id=task_id, result=result)
                
                # BUG-SHUTDOWN-001 FIX: Track fire-and-forget task
                bg_task = asyncio.create_task(
                    self._safe_record_routing_event(self._project_id, task_id, task.type, result)
                )
                self._background_tasks.add(bg_task)
                bg_task.add_done_callback(self._background_tasks.discard)

                # Improvement 13: write task output immediately after completion
                if _progress_writer is not None:
                    await _progress_writer.task_completed(task_id, result, task)

                for phase in ("generation", "cross_review", "evaluation"):
                    self._check_phase_budget(phase)

        for level_idx, level in enumerate(levels):
            logger.info(f"Processing level {level_idx}: {level}")
            if not self.budget.can_afford(0.01):
                logger.warning("Budget exhausted, halting before level %d", level_idx)
                break
            if not self.budget.time_remaining():
                logger.warning("Time limit reached, halting before level %d", level_idx)
                break

            # Filter tasks with unmet or failed dependencies.
            # A task is runnable only if ALL its dependencies completed or degraded.
            # If any dependency FAILED, downstream tasks are skipped — executing
            # them with missing/invalid context would propagate garbage output.
            runnable = []
            for task_id in level:
                # BUG-RACE-002 FIX: Protect results access with lock
                async with self._results_lock:
                    dep_results = [
                        self.results.get(dep, TaskResult(dep, "", 0.0, Model.GPT_4O_MINI))
                        for dep in tasks[task_id].dependencies
                    ]
                any_failed = any(r.status == TaskStatus.FAILED for r in dep_results)
                all_finished = all(
                    r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED, TaskStatus.FAILED)
                    for r in dep_results
                )
                if any_failed:
                    failed_deps = [
                        r.task_id for r in dep_results
                        if r.status == TaskStatus.FAILED
                    ]
                    logger.warning(
                        f"Skipping {task_id}: dependencies failed: {failed_deps}"
                    )
                    async with self._results_lock:
                        self.results[task_id] = TaskResult(
                            task_id=task_id, output="", score=0.0,
                            model_used=Model.GPT_4O_MINI,
                            status=TaskStatus.FAILED,
                        )
                elif all_finished:
                    runnable.append(task_id)
                else:
                    logger.warning(f"Skipping {task_id}: unmet dependencies")
                    async with self._results_lock:
                        self.results[task_id] = TaskResult(
                            task_id=task_id, output="", score=0.0,
                            model_used=Model.GPT_4O_MINI,
                            status=TaskStatus.FAILED,
                        )

            if not runnable:
                continue

            parallel_count = len(runnable)
            if parallel_count > 1:
                logger.info(
                    "Executing level %d: %d tasks in parallel (max=%d): %s",
                    level_idx, parallel_count, self._max_parallel_tasks, runnable,
                )
            else:
                logger.info("Executing level %d: %s", level_idx, runnable)

            logger.debug(f"About to run tasks: {runnable}")
            try:
                await asyncio.gather(*(_run_one(tid) for tid in runnable))
                logger.debug(f"Completed level {level_idx}")
            except Exception as e:
                logger.error(f"Level {level_idx} failed: {e}")
                # Continue with next level anyway

            # Checkpoint after each level completes
            state = self._make_state(project_desc, success_criteria, tasks,
                                     execution_order=execution_order)
            if runnable:
                await self.state_mgr.save_checkpoint(
                    self._project_id, runnable[-1], state
                )

        return self._make_state(project_desc, success_criteria, tasks,
                                execution_order=execution_order)

    def _extract_function_name(self, code: str) -> Optional[str]:
        """
        Extract the main function name from generated code.
        
        Args:
            code: Python source code
        
        Returns:
            Function name or None
        """
        import ast
        import re
        
        try:
            # Try AST parsing first
            tree = ast.parse(code)
            
            # Look for the first function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip dunder methods
                    if not node.name.startswith('__'):
                        return node.name
            
            # Fallback: Look for class __init__
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
                    
        except SyntaxError:
            # AST parsing failed, try regex
            match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            if match:
                return match.group(1)
            
            # Try class name
            class_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            if class_match:
                return class_match.group(1)
        
        return None

    def _build_delta_prompt(self, original_prompt: str, record: "AttemptRecord") -> str:
        """
        Build an enriched prompt for the next iteration after a failed attempt.

        Uses XML-style tags to delimit the feedback block so that adversarial
        LLM output cannot escape the block with injected plain-text sentinels
        or tag sequences.  All user-controlled fields are sanitised before
        embedding to prevent prompt-injection via failure_reason, validator
        names, or output_snippet.
        """

        def _sanitize(text: str) -> str:
            """Strip XML delimiters and the plain sentinel from user-supplied data."""
            # Remove our own XML tags so injected copies cannot break the structure.
            text = text.replace("<ORCHESTRATOR_FEEDBACK>", "")
            text = text.replace("</ORCHESTRATOR_FEEDBACK>", "")
            # Neutralise the plain sentinel so it cannot appear twice — once
            # legitimate and once injected — which would confuse the next model.
            text = text.replace("PREVIOUS ATTEMPT FAILED:", "[PREVIOUS ATTEMPT]:")
            return text

        safe_reason     = _sanitize(record.failure_reason)
        safe_validators = [_sanitize(v) for v in record.validators_failed]
        validators_str  = ", ".join(safe_validators) if safe_validators else "none"

        snippet_section = ""
        if record.output_snippet:
            safe_snippet    = _sanitize(record.output_snippet)
            snippet_section = f"\n- Output snippet: {safe_snippet}"

        return (
            f"{original_prompt}\n\n"
            f"<ORCHESTRATOR_FEEDBACK>\n"
            f"PREVIOUS ATTEMPT FAILED:\n"
            f"- Attempt: {record.attempt_num}\n"
            f"- Model: {record.model_used}\n"
            f"- Reason: {safe_reason}\n"
            f"- Validators failed: {validators_str}"
            f"{snippet_section}\n\n"
            f"Please correct specifically: {safe_reason}\n"
            f"</ORCHESTRATOR_FEEDBACK>"
        )

    async def _execute_task(self, task: Task) -> TaskResult:
        """
        Core loop: generate → critique → revise → evaluate
        With plateau detection, deterministic validation, and mid-task budget checks.
        """
        with traced_task(task.id, task.type.value) as span:
            span.set_attribute("task.description", task.prompt[:200])
            models = self._get_available_models(task.type)
            if not models:
                return TaskResult(
                    task_id=task.id, output="", score=0.0,
                    model_used=Model.GPT_4O_MINI,
                    status=TaskStatus.FAILED,
                )

            primary = models[0]
            reviewer = self._select_reviewer(primary, task.type)

            # Emit TaskStarted streaming event
            if self._event_bus:
                from .events import TaskStartedEvent
                await self._event_bus.publish(TaskStartedEvent(
                    aggregate_id=task.id,
                    task_id=task.id,
                    task_type=task.type.value,
                ))

            # Fire MODEL_SELECTED event so hooks can observe routing decisions
            self._hook_registry.fire(
                EventType.MODEL_SELECTED,
                task_id=task.id,
                model=primary.value,
                backend=get_provider(primary),
            )

            context = self._gather_dependency_context(task.dependencies)
            
            # OPTIMIZATION: Check multi-level cache (L1/L2/L3) for high-quality patterns
            cached_result = None
            if self._cache_optimizer and not context:
                cached_result = await self._cache_optimizer.get(
                    model=primary.value,
                    prompt=task.prompt,
                    max_tokens=task.max_output_tokens,
                    task_type=task.type,
                )
            
            # Fallback to old semantic cache if CacheOptimizer not available
            if cached_result is None and not context:
                cached_output = self._semantic_cache.get_cached_pattern(task)
                if cached_output:
                    cached_result = {
                        "response": cached_output,
                        "tokens_input": 0,
                        "tokens_output": 0,
                        "cost": 0.0,
                        "cached": True,
                    }
            
            if cached_result and not context:
                # Only use cache if no dependency context (context would make it different)
                cache_level = "L3" if cached_result.get("semantic") else "L1/L2"
                logger.info(f"  {task.id}: cache hit ({cache_level}), skipping generation")
                return TaskResult(
                    task_id=task.id,
                    output=cached_result["response"],
                    score=0.85,  # Cached patterns meet quality threshold
                    model_used=primary,
                    reviewer_model=None,
                    tokens_used={
                        "input": cached_result.get("tokens_input", 0),
                        "output": cached_result.get("tokens_output", 0)
                    },
                    iterations=0,
                    cost_usd=cached_result.get("cost", 0.0),
                    status=TaskStatus.COMPLETED,
                    critique="",
                    deterministic_check_passed=True,
                    degraded_fallback_count=0,
                    attempt_history=[],
                )
            
            full_prompt = task.prompt
            if context:
                # For code_review tasks, the dependency context IS the source code
                # being reviewed. Make this explicit so the LLM doesn't claim
                # "source code was not provided".
                if task.type == TaskType.CODE_REVIEW:
                    full_prompt += (
                        f"\n\n--- SOURCE CODE TO REVIEW (from prior tasks) ---\n"
                        f"The following is the actual generated source code you must "
                        f"review. Do NOT claim the code was not provided.\n\n"
                        f"{context}"
                    )
                else:
                    full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"

            best_output = ""
            best_score = 0.0
            best_critique = ""
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            degraded_count = 0
            scores_history: list[float] = []
            det_passed = True  # default: no validators = passed
            attempt_history: list[AttemptRecord] = []
            _failed_validator_names: list[str] = []
            model_escalated = False  # Track if we've tried model escalation

            logger.info(f"Executing {task.id} ({task.type.value}): "
                         f"primary={primary.value}, reviewer={reviewer.value if reviewer else 'none'}")

            for iteration in range(task.max_iterations):
                # FIX #5: Mid-task budget check — estimate minimum cost for one
                # generate+critique+revise+evaluate cycle (~0.02 USD min)
                if not self.budget.can_afford(0.02):
                    logger.warning(
                        f"Budget insufficient mid-task for {task.id} "
                        f"at iteration {iteration} "
                        f"(remaining: ${self.budget.remaining_usd:.4f})"
                    )
                    break

                if not self.budget.time_remaining():
                    logger.warning(f"Time limit reached mid-task for {task.id}")
                    break

                # ── GENERATE ──
                # DeepSeek-R1 is a chain-of-thought reasoning model whose
                # internal reasoning tokens count against max_tokens but don't appear in
                # content output. For code tasks on these models, double the token budget
                # (cap at 16384) to ensure complete output. Both also need longer timeouts.
                _provider = get_provider(primary)
                _is_reasoning_model = (
                    _provider == "anthropic" or  # Claude models can be reasoning-heavy
                    (_provider == "deepseek" and primary.value == "deepseek-reasoner")
                )
                _is_deepseek_chat = primary.value == "deepseek-chat"
                if _is_reasoning_model:
                    gen_timeout = 240
                    if task.type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
                        # Double token budget: reasoning tokens eat into output budget
                        effective_max_tokens = min(task.max_output_tokens * 2, 16384)
                    else:
                        effective_max_tokens = task.max_output_tokens
                elif _is_deepseek_chat:
                    # DeepSeek-Coder is slow but good - give it more time
                    gen_timeout = 180
                    effective_max_tokens = task.max_output_tokens
                elif task.type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
                    gen_timeout = 120
                    effective_max_tokens = task.max_output_tokens
                else:
                    gen_timeout = 60
                    effective_max_tokens = task.max_output_tokens
                
                # Apply model-specific token limits (e.g., Claude Haiku max 4096)
                model_limit = MODEL_MAX_TOKENS.get(primary)
                if model_limit:
                    effective_max_tokens = min(effective_max_tokens, model_limit)
                
                try:
                    # Build system prompt with explicit instructions for code tasks
                    system_prompt = f"You are an expert executing a {task.type.value} task. " \
                                   f"Produce high-quality, complete output."
                    if task.type == TaskType.CODE_GEN:
                        system_prompt += (
                            "\n\nCRITICAL REQUIREMENTS:\n"
                            "1. Return ONLY raw code - NO markdown fences (```), NO explanations outside code\n"
                            "2. Code must be THOROUGHLY COMMENTED - every function, class, and complex logic block\n"
                            "3. EVERY file MUST include this header comment using Python docstrings:\n"
                            "   \"\"\"\n"
                            "   Author: Georgios-Chrysovalantis Chatzivantsidis\n"
                            "   Description: [Brief description of the file's purpose]\n"
                            "   \"\"\"\n"
                            "4. Output must be valid, complete, production-ready code that passes syntax checks.\n"
                            "5. Use Python-style comments (# or docstrings), NEVER C-style comments (/* */)"
                        )
                    
                    # Use temperature 0.0 for code generation (deterministic output)
                    # DeepSeek recommends 0.0 for coding/math tasks.
                    # Note: deepseek-reasoner ignores temperature (API handles this)
                    gen_temperature = 0.0 if task.type == TaskType.CODE_GEN else 0.3
                    
                    logger.info(f"  {task.id}: calling {primary.value} for generation (timeout={gen_timeout}s)")
                    gen_response = await self.client.call(
                        primary, full_prompt,
                        system=system_prompt,
                        max_tokens=effective_max_tokens,
                        temperature=gen_temperature,
                        timeout=gen_timeout,
                    )
                    logger.info(f"  {task.id}: generation complete, tokens={gen_response.input_tokens}/{gen_response.output_tokens}")
                    output = _clean_code_output(gen_response.text, task.type)
                    gen_cost = gen_response.cost_usd
                    self.budget.charge(gen_cost, "generation")
                    if self._cost_predictor is not None:
                        self._cost_predictor.record(primary, task.type, gen_cost)
                    total_cost += gen_cost
                    total_input_tokens += gen_response.input_tokens
                    total_output_tokens += gen_response.output_tokens
                    self._record_success(primary, gen_response)
                except (Exception, asyncio.CancelledError) as e:
                    logger.error(f"Generation failed for {task.id}: {e}")
                    self._record_failure(primary, error=e)
                    degraded_count += 1
                    
                    # OPTIMIZATION: Escalate tier on failure
                    self._escalate_tier(task.type)
                    
                    fb = self._get_fallback(primary)
                    if fb:
                        try:
                            # Reuse same system prompt as primary
                            fb_system = f"You are an expert executing a {task.type.value} task. " \
                                       f"Produce high-quality, complete output."
                            if task.type == TaskType.CODE_GEN:
                                fb_system += (
                                    "\n\nCRITICAL REQUIREMENTS:\n"
                                    "1. Return ONLY raw code - NO markdown fences (```), NO explanations outside code\n"
                                    "2. Code must be THOROUGHLY COMMENTED - every function, class, and complex logic block\n"
                                    "3. EVERY file MUST include this header comment using Python docstrings:\n"
                                    "   \"\"\"\n"
                                    "   Author: Georgios-Chrysovalantis Chatzivantsidis\n"
                                    "   Description: [Brief description of the file's purpose]\n"
                                    "   \"\"\"\n"
                                    "4. Output must be valid, complete, production-ready code that passes syntax checks.\n"
                                    "5. Use Python-style comments (# or docstrings), NEVER C-style comments (/* */)"
                                )
                            # Use temperature 0.0 for code generation (deterministic output)
                            fb_temperature = 0.0 if task.type == TaskType.CODE_GEN else 0.3
                            
                            gen_response = await self.client.call(
                                fb, full_prompt,
                                system=fb_system,
                                max_tokens=effective_max_tokens,
                                temperature=fb_temperature,
                                timeout=gen_timeout,
                            )
                            output = _clean_code_output(gen_response.text, task.type)
                            self.budget.charge(gen_response.cost_usd, "generation")
                            total_cost += gen_response.cost_usd
                            total_input_tokens += gen_response.input_tokens
                            total_output_tokens += gen_response.output_tokens
                            self._record_success(fb, gen_response)
                            primary = fb
                        except (Exception, asyncio.CancelledError) as e2:
                            logger.error(f"Fallback generation also failed: {e2}")
                            self._record_failure(fb, error=e2)
                            break
                    else:
                        break

                # FIX #5: Re-check budget after generation before critique
                if not self.budget.can_afford(0.005):
                    logger.warning(f"Budget depleted after generation for {task.id}")
                    scores_history.append(0.0)
                    if not best_output:
                        best_output = output
                    break

                # ── CRITIQUE (cross-model) ──
                critique = ""
                if reviewer and reviewer != primary:
                    # Reviewer token budget: reasoning models (Claude, DeepSeek-R1)
                    # consume their token budget on internal chain-of-thought, so they
                    # need the same doubled budget as when generating. Standard models
                    # only need 800 tokens to produce a focused critique.
                    _rev_provider = get_provider(reviewer)
                    _reviewer_is_reasoning = (
                        _rev_provider == "anthropic" or  # Claude models
                        (_rev_provider == "deepseek" and reviewer.value == "deepseek-reasoner")
                    )
                    if _reviewer_is_reasoning:
                        critique_max_tokens = min(task.max_output_tokens * 2, 8192)
                        critique_timeout = 240
                    else:
                        critique_max_tokens = 1200  # raised: 800 was too low for detailed reviews
                        critique_timeout = 60
                    
                    # Apply model-specific token limits for reviewer
                    reviewer_limit = MODEL_MAX_TOKENS.get(reviewer)
                    if reviewer_limit:
                        critique_max_tokens = min(critique_max_tokens, reviewer_limit)
                    
                    try:
                        # Use low temperature for focused, deterministic critique
                        critique_temperature = 0.2 if task.type == TaskType.CODE_GEN else 0.3
                        
                        critique_response = await self.client.call(
                            reviewer,
                            f"Review this output for correctness, completeness, and quality. "
                            f"Be specific about flaws and suggest concrete improvements.\n\n"
                            f"ORIGINAL TASK: {task.prompt}\n\n"
                            f"OUTPUT TO REVIEW:\n{output}",
                            system="You are a critical reviewer. Find flaws, be specific.",
                            max_tokens=critique_max_tokens,
                            temperature=critique_temperature,
                            timeout=critique_timeout,
                        )
                        critique = critique_response.text
                        self.budget.charge(critique_response.cost_usd, "cross_review")
                        if self._cost_predictor is not None:
                            self._cost_predictor.record(primary, task.type, critique_response.cost_usd)
                        total_cost += critique_response.cost_usd
                        # Record success to reset circuit breaker counter for reviewer.
                        # This ensures counter only tracks consecutive failures, allowing recovery
                        # from transient errors between successful critiques.
                        self._record_success(reviewer, critique_response)
                    except (Exception, asyncio.CancelledError) as e:
                        logger.warning(f"Critique failed for {task.id}: {e}")
                        # Use _record_failure() for graduated circuit breaker instead of immediate kill.
                        # This allows transient errors (429, timeout) to be retried; only permanent
                        # errors (401, 404) or 3 consecutive failures disable the model.
                        self._record_failure(reviewer, error=e)
                        degraded_count += 1

                # ── REVISE (if critique exists) ──
                # Skip revision for reasoning models (Claude, DeepSeek-R1): their
                # chain-of-thought makes revision calls as slow/expensive as generation.
                # Instead, embed the critique into the next iteration's prompt so the
                # model can self-correct on re-generation.
                if critique and not _is_reasoning_model:
                    try:
                        revise_response = await self.client.call(
                            primary,
                            f"Revise your output based on this critique. "
                            f"Address every specific issue raised.\n\n"
                            f"ORIGINAL TASK: {task.prompt}\n\n"
                            f"YOUR OUTPUT:\n{output}\n\n"
                            f"CRITIQUE:\n{critique}\n\n"
                            f"Produce the complete improved version.",
                            system=f"You are revising a {task.type.value} task based on peer review.",
                            max_tokens=effective_max_tokens,
                            timeout=gen_timeout,
                        )
                        output = revise_response.text
                        self.budget.charge(revise_response.cost_usd, "generation")
                        total_cost += revise_response.cost_usd
                    except (Exception, asyncio.CancelledError) as e:
                        logger.warning(f"Revision failed for {task.id}: {e}")
                elif critique and _is_reasoning_model:
                    # Embed critique into the next iteration's prompt for self-correction.
                    full_prompt = (
                        f"{task.prompt}\n\n"
                        f"--- PEER REVIEW FEEDBACK (incorporate in your response) ---\n"
                        f"{critique}\n"
                        f"--- END FEEDBACK ---"
                    )
                    if context:
                        full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"
                    logger.debug(
                        f"{primary.value}: critique embedded into next iteration "
                        f"prompt for {task.id}"
                    )

                # ── DETERMINISTIC VALIDATION ──
                det_passed = True
                # Filter validators based on task content type
                validators = self._filter_validators_for_task(task, output)
                if validators:
                    val_results = await async_run_validators(output, validators)
                    det_passed = all_validators_pass(val_results)
                    if not det_passed:
                        failed = [v for v in val_results if not v.passed]
                        _failed_validator_names = [v.validator_name for v in failed]
                        logger.warning(
                            f"Deterministic check failed for {task.id}: "
                            f"{[f'{v.validator_name}: {v.details}' for v in failed]}"
                        )
                        # Record validator failure in telemetry + fire hook (Improvement 3)
                        self._telemetry.record_validator_failure(primary)
                        self._hook_registry.fire(
                            EventType.VALIDATION_FAILED,
                            task_id=task.id,
                            model=primary.value,
                            validators=validators,
                        )

                # ── EVALUATE ──
                if det_passed:
                    logger.info(f"  {task.id}: starting evaluation...")
                    score = await self._evaluate(task, output)
                    logger.info(f"  {task.id}: evaluation complete, score={score:.3f}")
                else:
                    score = 0.0
                    logger.info(f"  {task.id}: skipped evaluation (det check failed)")

                self.budget.charge(0.0, "evaluation")
                scores_history.append(score)

                if score > best_score:
                    best_output = output
                    best_score = score
                    best_critique = critique

                # Emit TaskProgressUpdate streaming event
                if self._event_bus:
                    try:
                        from .unified_events import TaskProgressEvent
                        await self._event_bus.publish(TaskProgressEvent(
                            aggregate_id=task.id,
                            task_id=task.id,
                            iteration=iteration + 1,
                            score=score,
                            message=f"Best score: {best_score:.3f}",
                        ))
                    except ImportError:
                        # Fallback to standard events
                        from .events import TaskProgressEvent
                        await self._event_bus.publish(TaskProgressEvent(
                            task_id=task.id,
                            iteration=iteration + 1,
                            score=score,
                            best_score=best_score,
                            model=primary.value,
                        ))
                
                # Notify dashboard of task progress
                self._notify_dashboard_task_progress(iteration + 1, best_score)

                logger.info(
                    f"  {task.id} iter {iteration + 1}: score={score:.3f} "
                    f"(best={best_score:.3f}, threshold={task.acceptance_threshold})"
                )

                # ── FAILURE HISTORY + DELTA-PROMPT (Improvement 8) ──
                _iteration_passed = (score >= task.acceptance_threshold and det_passed)
                if not _iteration_passed:
                    if not det_passed:
                        _failure_reason = (
                            f"Deterministic check failed: validators={_failed_validator_names}"
                        )
                    else:
                        _failure_reason = (
                            f"Score {score:.3f} below threshold {task.acceptance_threshold}"
                        )
                    _record = AttemptRecord(
                        attempt_num=iteration + 1,
                        model_used=primary.value,
                        output_snippet=output[:200],
                        failure_reason=_failure_reason,
                        validators_failed=list(_failed_validator_names),
                    )
                    attempt_history.append(_record)
                    self._hook_registry.fire(
                        EventType.TASK_RETRY_WITH_HISTORY,
                        task_id=task.id,
                        attempt_num=iteration + 1,
                        record=_record,
                    )
                    # Build delta-prompt for the next iteration (not the last)
                    if iteration < task.max_iterations - 1:
                        full_prompt = self._build_delta_prompt(task.prompt, _record)
                        if context:
                            full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"

                # Reset per-iteration validator names
                _failed_validator_names = []

                # ── CONVERGENCE CHECKS ──
                if best_score >= task.acceptance_threshold:
                    logger.info(f"  {task.id}: threshold met at iteration {iteration + 1}")
                    break

                # OPTIMIZATION: Confidence-based early exit
                # Exit early if we've seen stable high performance across recent iterations
                if self._should_exit_early(scores_history, task.acceptance_threshold):
                    logger.info(
                        f"  {task.id}: early exit due to stable high performance "
                        f"(confidence window met)"
                    )
                    break

                if len(scores_history) >= 2:
                    delta = abs(scores_history[-1] - scores_history[-2])
                    if delta < 0.02:
                        # Only stop on plateau if we have a usable score.
                        # If best_score is still below half the acceptance threshold,
                        # keep trying — the critique/revision cycle may still help.
                        if best_score >= task.acceptance_threshold * 0.5:
                            logger.info(f"  {task.id}: plateau detected (Δ={delta:.4f})")
                            break
                        elif len(scores_history) >= 3:
                            # After 3+ iterations with no improvement AND bad score,
                            # try model escalation before giving up
                            if best_score >= task.acceptance_threshold * 0.3 and not model_escalated:
                                # Try escalating to a better model
                                next_model = self._get_next_tier_model(primary, task.type)
                                if next_model and self.budget.can_afford(0.05):
                                    logger.info(
                                        f"  {task.id}: plateau at low score (best={best_score:.3f}), "
                                        f"escalating from {primary.value} to {next_model.value}"
                                    )
                                    primary = next_model
                                    model_escalated = True
                                    # Keep best_output as context for warm start
                                    full_prompt = (
                                        f"{task.prompt}\n\n"
                                        f"--- PREVIOUS ATTEMPT (score: {best_score:.2f}) ---\n"
                                        f"This is a previous attempt that needs improvement:\n"
                                        f"{best_output}\n\n"
                                        f"--- YOUR TASK ---\n"
                                        f"Improve upon the previous attempt to achieve a higher quality score. "
                                        f"Focus on fixing issues and enhancing the solution."
                                    )
                                    continue  # Continue loop with new model
                            
                            # Give up to avoid wasting budget
                            logger.info(
                                f"  {task.id}: plateau at low score after "
                                f"{len(scores_history)} iters (Δ={delta:.4f}, "
                                f"best={best_score:.3f})"
                            )
                            break

            status = TaskStatus.COMPLETED if best_score >= task.acceptance_threshold else TaskStatus.DEGRADED
            if best_score == 0.0 and not det_passed:
                status = TaskStatus.FAILED

            # Feed final eval score back to telemetry so ConstraintPlanner re-ranks
            if best_score > 0.0:
                self._telemetry.record_call(
                    primary, latency_ms=0.0, cost_usd=0.0,
                    success=(status != TaskStatus.FAILED),
                    quality_score=best_score,
                )
            
            # OPTIMIZATION: Cache successful patterns for reuse (L1/L2/L3)
            if status == TaskStatus.COMPLETED and best_output:
                # L3: Semantic cache
                self._semantic_cache.cache_pattern(task, best_output, best_score)
                
                # L1/L2: Cache optimizer (with compression and TTL)
                if self._cache_optimizer:
                    await self._cache_optimizer.put(
                        model=primary.value,
                        prompt=task.prompt,
                        max_tokens=task.max_output_tokens,
                        response=best_output,
                        tokens_input=total_input_tokens,
                        tokens_output=total_output_tokens,
                        cost=total_cost,
                        task_type=task.type,
                        quality_score=best_score,
                    )

            # Emit TaskCompleted or TaskFailed streaming event
            if self._event_bus:
                from .events import TaskCompletedEvent, TaskFailedEvent
                if status == TaskStatus.FAILED:
                    await self._event_bus.publish(TaskFailedEvent(
                        aggregate_id=task.id,
                        task_id=task.id,
                        error="all attempts failed",
                    ))
                else:
                    await self._event_bus.publish(TaskCompletedEvent(
                        aggregate_id=task.id,
                        task_id=task.id,
                        score=best_score,
                        model=primary.value,
                        cost_usd=total_cost,
                        iterations=len(scores_history),
                        status=status.value,
                    ))

            span.set_attribute("task.status", status.value)
            span.set_attribute("task.score", best_score or 0.0)
            
            # Save to progressive output structure (bmalph-style)
            saved_files = []
            if '_prog_output' in locals() and _prog_output is not None:
                saved_path = _prog_output.save_task_output(
                    task=task,
                    output=best_output,
                    model=primary.value,
                    cost_usd=total_cost,
                    tokens_input=total_input_tokens,
                    tokens_output=total_output_tokens,
                    score=best_score or 0.0,
                    status=status,
                )
                saved_files = [saved_path] if saved_path else []
            
            # Git auto-commit after task (bmalph-style TDD commits)
            if (self._git_integration is not None and
                self._git_integration.is_available() and
                saved_files):
                try:
                    commit_hash = self._git_integration.commit_task_completion(
                        task_id=task.id,
                        status=status.value,
                        model=primary.value,
                        cost=total_cost,
                        score=best_score or 0.0,
                        files=saved_files,
                    )
                    if commit_hash:
                        logger.info(f"  {task.id}: git commit {commit_hash}")
                except Exception as e:
                    logger.warning(f"  {task.id}: git commit failed: {e}")

            # NEW: Test validation for code generation tasks
            if (HAS_TEST_VALIDATOR and
                task.type == TaskType.CODE_GEN and
                best_output and
                hasattr(self, '_output_dir') and
                self._output_dir):

                logger.info(f"  {task.id}: Validating test generation...")
                try:
                    # FIRST: Validate syntax of generated code
                    try:
                        compile(best_output, '<generated>', 'exec')
                        logger.debug(f"  {task.id}: Generated code syntax OK")
                    except SyntaxError as e:
                        logger.warning(f"  {task.id}: Generated code has syntax error: {e}")
                        # Don't proceed with test validation if code is invalid
                        best_output = None  # Mark as invalid
                        status = TaskStatus.FAILED
                        best_score = 0.0
                    
                    # Find the generated source file
                    source_file = None
                    if self._output_dir.exists() and best_output:
                        for f in self._output_dir.rglob("*.py"):
                            if task.id.replace('_', '') in f.stem.replace('_', ''):
                                source_file = f
                                break

                    if source_file and source_file.exists() and best_output:
                        # Extract function name from output
                        func_name = self._extract_function_name(best_output)

                        # Validate test generation
                        validator = TestValidator(max_iterations=2)
                        test_result = await validator.validate_test_generation(
                            source_file=source_file,
                            function_name=func_name or "main",
                            project_root=self._output_dir,
                        )

                        if test_result.passed:
                            # Save the validated test
                            test_file = source_file.parent / f"test_{source_file.name}"
                            test_file.write_text(test_result.test_code, encoding='utf-8')
                            logger.info(f"  {task.id}: ✅ Test validated and saved: {test_file.name}")
                        else:
                            logger.warning(f"  {task.id}: ⚠️ Test validation failed: {test_result.error_message[:200]}")
                    else:
                        logger.debug(f"  {task.id}: No source file found for test validation")

                except Exception as e:
                    logger.warning(f"  {task.id}: Test validation failed: {e}")

            return TaskResult(
                task_id=task.id,
                output=best_output,
                score=best_score,
                model_used=primary,
                reviewer_model=reviewer,
                tokens_used={"input": total_input_tokens, "output": total_output_tokens},
                iterations=len(scores_history),
                cost_usd=total_cost,
                status=status,
                critique=best_critique,
                deterministic_check_passed=det_passed,
                degraded_fallback_count=degraded_count,
                attempt_history=attempt_history,
            )

    async def _run_preflight_check(
        self,
        task: "Task",
        output: str,
        score: float,
        primary: "Model",
    ) -> "tuple[str, float, PreflightResult]":
        """
        Post-loop preflight delivery gate.

        Checks best_output before finalizing TaskResult:
        - PASS  : return unchanged
        - WARN  : log + score * 0.85, fire PREFLIGHT_CHECK hook
        - ENRICH: 1 extra LLM revision with enrich reason as critique
        - BLOCK : 1 extra LLM revision with block reason as critique
                     -> recovered: return revised output
                     -> still BLOCK: return original output, score=0.0

        Fail-open: any validator exception is caught and treated as PASS.
        """
        from .preflight import PreflightAction, PreflightMode, PreflightResult

        try:
            pf_result = self._preflight_validator.validate(
                response=output,
                context={
                    "task_type": task.type.value,
                    "user_request": task.prompt[:200],
                    "model": primary.value,
                    "score": score,
                },
                mode=PreflightMode.AUTO,
            )
        except Exception as exc:
            logger.warning("preflight validator raised: %s — treating as PASS", exc)
            return output, score, PreflightResult(action=PreflightAction.PASS, passed=True)

        if pf_result.action == PreflightAction.PASS:
            return output, score, pf_result

        if pf_result.action == PreflightAction.WARN:
            penalized = round(score * 0.85, 4)
            logger.warning(
                "[preflight] WARN task=%s score %.3f->%.3f: %s",
                task.id, score, penalized, "; ".join(pf_result.warnings),
            )
            self._hook_registry.fire(
                EventType.PREFLIGHT_CHECK,
                task_id=task.id,
                action="warn",
                reason="; ".join(pf_result.warnings),
                score_before=score,
                score_after=penalized,
            )
            return output, penalized, pf_result

        # ENRICH or BLOCK — attempt one extra revision
        critique_text = pf_result.reason or pf_result.enrichment or "Improve the response quality."
        logger.info(
            "[preflight] %s task=%s — attempting 1 revision: %s",
            pf_result.action.value.upper(), task.id, critique_text[:100],
        )
        try:
            revised_prompt = (
                f"{task.prompt}\n\n"
                f"[Revision required] {critique_text}\n"
                f"Please revise your previous response to address the above."
            )
            gen_response = await self.client.call(
                primary,
                revised_prompt,
                system=f"You are an expert executing a {task.type.value} task. Produce high-quality, complete output.",
                max_tokens=task.max_output_tokens,
                temperature=0.3,
                timeout=120,
            )
            revised_output = gen_response.text
        except Exception as exc:
            logger.warning("[preflight] revision LLM call failed (%s) — using original", exc)
            self._hook_registry.fire(
                EventType.PREFLIGHT_CHECK,
                task_id=task.id,
                action=pf_result.action.value + "_revision_failed",
                reason=str(exc),
                score_before=score,
                score_after=score,
            )
            return output, score, pf_result

        # Re-validate the revised output
        try:
            retry_result = self._preflight_validator.validate(
                response=revised_output,
                context={"task_type": task.type.value, "user_request": task.prompt[:200]},
                mode=PreflightMode.AUTO,
            )
        except Exception:
            retry_result = pf_result  # treat same as original if validator fails

        if retry_result.action == PreflightAction.BLOCK:
            logger.warning("[preflight] BLOCK task=%s — revision still blocked, score->0", task.id)
            self._hook_registry.fire(
                EventType.PREFLIGHT_CHECK,
                task_id=task.id,
                action="block_degraded",
                reason=retry_result.reason or "Still blocked after revision",
                score_before=score,
                score_after=0.0,
            )
            return output, 0.0, retry_result

        logger.info("[preflight] %s recovered task=%s", pf_result.action.value.upper(), task.id)
        self._hook_registry.fire(
            EventType.PREFLIGHT_CHECK,
            task_id=task.id,
            action=pf_result.action.value + "_recovered",
            reason=critique_text[:100],
            score_before=score,
            score_after=score,
        )
        return revised_output, score, retry_result

    async def _evaluate(self, task: Task, output: str) -> float:
        """LLM-based scoring with self-consistency (2 runs, Δ ≤ 0.05)."""
        eval_models = self._get_available_models(TaskType.EVALUATE)
        if not eval_models:
            logger.debug(f"  {task.id}: no eval models available, returning 0.5")
            return 0.5

        eval_model = eval_models[0]
        logger.debug(f"  {task.id}: evaluating with {eval_model.value}")
        
        eval_prompt = (
            f"Score this output on a scale of 0.0 to 1.0.\n"
            f"Evaluate: correctness, completeness, quality, adherence to task.\n\n"
            f"TASK: {task.prompt}\n"
            f"ACCEPTANCE THRESHOLD: {task.acceptance_threshold}\n\n"
            f"OUTPUT:\n{output}\n\n"
            f'Return ONLY JSON: {{"score": <float>, "reasoning": "<brief>"}}'
        )

        scores = []
        for run in range(2):
            try:
                logger.debug(f"  {task.id}: eval run {run + 1}/2 starting...")
                response = await self.client.call(
                    eval_model, eval_prompt,
                    system="You are a precise evaluator. Score exactly, return only JSON.",
                    max_tokens=300,
                    temperature=0.1,
                    timeout=60,
                )
                logger.debug(f"  {task.id}: eval run {run + 1}/2 complete, score={self._parse_score(response.text):.3f}")
                self.budget.charge(response.cost_usd, "evaluation")
                score = self._parse_score(response.text)
                scores.append(score)
            except (Exception, asyncio.CancelledError) as e:
                logger.warning(f"Evaluation run {run + 1} failed: {e}")
                scores.append(0.5)

        if len(scores) == 2:
            delta = abs(scores[0] - scores[1])
            if delta > 0.05:
                logger.warning(
                    f"Evaluation inconsistency: {scores[0]:.3f} vs {scores[1]:.3f} "
                    f"(Δ={delta:.3f} > 0.05). Using lower score."
                )
                return min(scores)
            return sum(scores) / len(scores)

        return scores[0] if scores else 0.5

    def _parse_score(self, text: str) -> float:
        text = text.strip()
        logger.debug(f"_parse_score: parsing text length={len(text)}")
        try:
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = json.loads(text.strip())
            score = float(data.get("score", 0.5))
            logger.debug(f"_parse_score: parsed score={score}")
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"_parse_score: JSON parse failed: {e}")
            pass
        match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if match:
            score = float(match.group(1))
            logger.debug(f"_parse_score: regex parsed score={score}")
            return max(0.0, min(1.0, score))
        logger.warning(f"Could not parse score from: {text[:100]}")
        return 0.5

    # ─────────────────────────────────────────
    # Telemetry + circuit breaker helpers
    # ─────────────────────────────────────────

    def _record_success(self, model: Model, response: "APIResponse") -> None:
        """Record a successful API call; reset circuit breaker counter."""
        self._consecutive_failures[model] = 0
        self._adaptive_router.record_success(model)
        self._adaptive_router.record_latency(model, response.latency_ms)
        
        # Notify dashboard of model success
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_model_success(model)
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")
        self._telemetry.record_call(
            model,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            success=True,
        )
        # Feed rate-limit tracker so _apply_filters can enforce sliding-window caps
        self._planner.rate_limit_tracker.record(
            provider=get_provider(model),
            cost_usd=response.cost_usd,
            tokens=response.input_tokens + response.output_tokens,
        )

    def _record_failure(self, model: Model, error: Optional[Exception] = None) -> None:
        """
        Record a failed API call. Increment circuit breaker counter.
        If consecutive failures reach the threshold, mark the model unhealthy.
        401 Unauthorized errors immediately mark the model unhealthy (permanent auth failure).
        """
        # Notify dashboard of model failure
        if self._dashboard_integration:
            try:
                self._dashboard_integration.on_model_failure(model)
            except Exception as e:
                logger.debug(f"Dashboard notification failed: {e}")
        
        # 401/404/400 = permanent failure — mark unhealthy immediately, no retries needed
        # 401 = bad API key, 404 = wrong model name, 400 = bad request (e.g. invalid param)
        error_str = str(error) if error else ""
        is_permanent_error = (
            "401" in error_str or "invalid_authentication" in error_str.lower()
            or "404" in error_str or "not found" in error_str.lower()
            or ("400" in error_str and "invalid_request_error" in error_str.lower())
        )
        if is_permanent_error:
            if self.api_health.get(model, True):
                self.api_health[model] = False
                if "401" in error_str:
                    reason = "auth error (401) — check your API key"
                elif "404" in error_str:
                    reason = "model not found (404) — check model name"
                else:
                    reason = f"invalid request (400) — {error_str[error_str.find('message'):error_str.find('message')+80]}"
                logger.warning(
                    f"Model {model.value} marked unhealthy immediately: {reason}."
                )
            # Task 6: auth/permanent errors permanently disable in adaptive router
            if "401" in error_str or "invalid_authentication" in error_str.lower():
                self._adaptive_router.record_auth_failure(model)
            self._telemetry.record_call(model, latency_ms=0.0, cost_usd=0.0, success=False)
            return

        self._consecutive_failures[model] = self._consecutive_failures.get(model, 0) + 1
        # Task 6: record timeout in adaptive router for degradation tracking
        _is_timeout = (
            "timeout" in error_str.lower() or "timed out" in error_str.lower()
            or "asyncio.timeouterror" in error_str.lower()
            or "TimeoutError" in (type(error).__name__ if error else "")
        )
        if _is_timeout:
            self._adaptive_router.record_timeout(model)
        self._telemetry.record_call(
            model, latency_ms=0.0, cost_usd=0.0, success=False
        )
        if self._consecutive_failures[model] >= self._CIRCUIT_BREAKER_THRESHOLD:
            if self.api_health.get(model, True):
                self.api_health[model] = False
                logger.warning(
                    f"Circuit breaker tripped for {model.value} "
                    f"after {self._consecutive_failures[model]} consecutive failures"
                )

    def _get_active_policies(self, task_id: str = "") -> list[Policy]:
        """Return merged global + node-level policies for the given task."""
        return self._active_policies.policies_for(task_id)

    def _should_exit_early(
        self,
        scores_history: list[float],
        threshold: float,
        confidence_window: int = 2,
        variance_tolerance: float = 0.001
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

    # OPTIMIZATION: Tiered model selection for cost efficiency
    # PRIORITY: Fast models first, then quality
    _TIER_CHEAP = [
        Model.MISTRAL_NEMO,         # $0.02/$0.04 - fastest capable
        Model.MISTRAL_SMALL_3_1,    # $0.03/$0.11 - excellent value
        Model.GPT_4O_MINI,          # $0.15/$0.60 - reliable
        Model.GEMINI_2_5_FLASH_LITE,# $0.10/$0.40 - fast
    ]
    _TIER_BALANCED = [
        Model.GPT_4O,               # $2.50/$10 - premium reliable
        Model.GEMINI_FLASH,         # $0.15/$0.60 - 1M context
        Model.CLAUDE_3_HAIKU,       # $0.50/$2.50 - Claude cheap tier
        # DEEPSEEK moved to fallback - too slow (180s+)
    ]
    _TIER_PREMIUM = [
        Model.CLAUDE_SONNET_4_6,    # $3/$15 - best coding
        Model.GPT_5,                # $5/$20 - GPT-5 series
        Model.GEMINI_PRO,           # $1.25/$10 - Gemini premium
        Model.O4_MINI,              # $1.50/$6.00 - reasoning
    ]
    
    # Track tier escalation per task type to prevent loops
    _tier_escalation_count: dict[str, int] = {}

    def _get_available_models(self, task_type: TaskType) -> list[Model]:
        """
        Get available models with tiered selection for cost optimization.
        
        Uses three-tier routing: CHEAP → BALANCED → PREMIUM
        Starts with cheaper models and escalates if needed based on
        task complexity and previous failures.
        """
        # Check if we should try cheap tier first
        tier_key = f"{task_type.value}"
        escalation_count = self._tier_escalation_count.get(tier_key, 0)
        
        # Determine which tier to use based on escalation history
        if escalation_count == 0:
            # Start with cheap tier for simple tasks
            if task_type in (TaskType.DATA_EXTRACT, TaskType.SUMMARIZE):
                candidates = self._TIER_CHEAP + self._TIER_BALANCED
            else:
                candidates = self._TIER_BALANCED + self._TIER_CHEAP
        elif escalation_count == 1:
            # Escalate to balanced/premium
            candidates = self._TIER_BALANCED + self._TIER_PREMIUM
        else:
            # Full escalation - use premium models
            candidates = ROUTING_TABLE.get(task_type, [])
        
        available = [m for m in candidates if self.api_health.get(m, False)]
        if not available:
            available = [m for m in Model if self.api_health.get(m, False)]
        # Task 6: also filter out models the adaptive router has degraded/disabled
        available = [m for m in available if self._adaptive_router.is_available(m)]
        
        return available
    
    def _escalate_tier(self, task_type: TaskType) -> None:
        """Escalate to higher tier after cheap tier failure."""
        tier_key = f"{task_type.value}"
        self._tier_escalation_count[tier_key] = self._tier_escalation_count.get(tier_key, 0) + 1
        logger.info(f"Tier escalation for {task_type.value}: level {self._tier_escalation_count[tier_key]}")

    def _get_fast_decomposition_model(self) -> Model:
        """Get a fast, reliable model for task decomposition.
        
        Prioritizes speed and reliability over cost for decomposition,
        since decomposition is a critical path and happens once per project.
        """
        from .models import Model
        
        # Fast, reliable models for decomposition (in priority order)
        fast_models = [
            Model.GPT_4O_MINI,      # Fast, reliable, $0.15/$0.60
            Model.GEMINI_FLASH,     # Fast, 1M context, $0.15/$0.60  
            Model.GEMINI_2_5_FLASH_LITE,  # $0.10/$0.40
            Model.MISTRAL_SMALL_3_1,      # $0.03/$0.11 - good if available
            Model.MISTRAL_NEMO,           # $0.02/$0.04 - cheapest capable
        ]
        
        for m in fast_models:
            if self.api_health.get(m, False):
                logger.debug(f"Using {m.value} for decomposition")
                return m
        
        # Fallback to cheapest available
        return self._get_cheapest_available()
    
    def _get_cheapest_available(self) -> Model:
        from .models import COST_TABLE
        healthy = [m for m in Model if self.api_health.get(m, False)]
        if not healthy:
            raise RuntimeError("No healthy models available")
        return min(healthy, key=lambda m: COST_TABLE[m]["output"])

    def _select_reviewer(self, generator: Model, task_type: TaskType) -> Optional[Model]:
        gen_provider = get_provider(generator)
        candidates = self._get_available_models(task_type)

        # Only consider healthy models
        for c in candidates:
            if get_provider(c) != gen_provider and self.api_health.get(c, False):
                return c

        for c in candidates:
            if c != generator and self.api_health.get(c, False):
                return c

        return None

    def _get_fallback(self, failed_model: Model) -> Optional[Model]:
        fb = FALLBACK_CHAIN.get(failed_model)
        if fb and self.api_health.get(fb, False):
            return fb
        for m in Model:
            if m != failed_model and self.api_health.get(m, False):
                return m
        return None

    def _get_next_tier_model(self, current_model: Model, task_type: TaskType) -> Optional[Model]:
        """
        Get a higher-tier model for escalation when plateau detected.
        
        Tiers: CHEAP → BALANCED → PREMIUM
        Returns next tier up, or None if already at premium or no healthy models.
        """
        from .models import COST_TABLE
        
        # Define model tiers (higher index = better quality)
        model_tiers: dict[Model, int] = {
            # Cheap tier
            Model.GEMINI_FLASH_LITE: 0,
            Model.GPT_4O_MINI: 0,
            # Balanced tier
            Model.GEMINI_FLASH: 1,
            Model.DEEPSEEK_CHAT: 1,
            Model.CLAUDE_3_HAIKU: 1,
            # Premium tier
            Model.GPT_4O: 2,
            Model.DEEPSEEK_REASONER: 2,
            Model.GEMINI_PRO: 2,
        }
        
        current_tier = model_tiers.get(current_model, 1)
        
        # Get all models one tier higher that are healthy
        candidates = [
            m for m in Model
            if model_tiers.get(m, 1) > current_tier
            and self.api_health.get(m, False)
            and m in self._get_available_models(task_type)  # Valid for this task type
        ]
        
        if not candidates:
            return None
        
        # Return the cheapest model from the next tier
        return min(candidates, key=lambda m: COST_TABLE[m]["output"])

    # ─────────────────────────────────────────
    # DAG & dependency management
    # ─────────────────────────────────────────

    def _topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """
        Kahn's algorithm with cycle detection.
        FIX #6: Uses deque for O(1) popleft instead of list.sort()+pop(0) O(n²).
        """
        in_degree = {tid: 0 for tid in tasks}
        graph = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        # Sort initial zero-degree nodes for determinism, then use deque
        queue = deque(sorted(tid for tid, deg in in_degree.items() if deg == 0))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            # Sort neighbors for deterministic ordering before extending
            newly_ready = []
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    newly_ready.append(neighbor)
            newly_ready.sort()
            queue.extend(newly_ready)

        if len(result) != len(tasks):
            cycle_tasks = set(tasks.keys()) - set(result)
            logger.error(f"Dependency cycle detected involving: {cycle_tasks}")
            return result

        return result

    def _topological_levels(self, tasks: dict[str, Task]) -> list[list[str]]:
        """
        Group tasks into execution levels using Kahn's algorithm.

        Tasks at the same level have no dependencies on each other and can be
        executed in parallel. Level 0 = tasks with no dependencies, Level 1 =
        tasks whose only dependencies are in Level 0, and so on.

        Returns a list of levels, each level being a sorted list of task IDs.
        The union of all levels equals the full topological order.
        """
        in_degree = {tid: 0 for tid in tasks}
        graph: dict[str, list[str]] = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        levels: list[list[str]] = []
        ready = sorted(tid for tid, deg in in_degree.items() if deg == 0)

        while ready:
            levels.append(ready)
            next_ready: list[str] = []
            for node in ready:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = sorted(next_ready)

        return levels

    def _filter_validators_for_task(self, task: Task, output: str) -> list[str]:
        """
        Filter validators based on task type and content.
        Removes Python-specific validators for non-Python tasks.
        """
        if not task.hard_validators:
            return []
        
        # Detect if this is a Python task
        is_python_task = (
            "python" in task.prompt.lower() or
            ".py" in task.target_path.lower() or
            "flask" in task.prompt.lower() or
            "django" in task.prompt.lower() or
            "fastapi" in task.prompt.lower() or
            "def " in output[:500] or  # Check output for Python function defs
            "import " in output[:500]   # Check output for Python imports
        )
        
        # Detect if this is a web task (HTML/CSS/JS)
        is_web_task = (
            "html" in task.prompt.lower() or
            "css" in task.prompt.lower() or
            "javascript" in task.prompt.lower() or
            "js" in task.prompt.lower() or
            ".html" in task.target_path.lower() or
            ".css" in task.target_path.lower() or
            ".js" in task.target_path.lower() or
            "<!DOCTYPE" in output[:100] or
            "<html" in output[:100] or
            "function(" in output[:500] or
            "const " in output[:500]
        )
        
        if is_web_task or not is_python_task:
            # Remove Python-specific validators
            original = set(task.hard_validators)
            filtered = [v for v in task.hard_validators if v not in ("python_syntax", "ruff", "pytest")]
            removed = original - set(filtered)
            if removed:
                logger.info(f"Task {task.id}: skipped Python validators {removed} (non-Python content detected)")
            return filtered
        
        return task.hard_validators

    def _gather_dependency_context(self, dep_ids: list[str]) -> str:
        """
        Gather output from completed/degraded dependencies.
        Truncates each dependency's output to self.context_truncation_limit chars
        and logs a warning when truncation occurs so information loss is visible.
        """
        parts = []
        limit = self.context_truncation_limit
        for dep_id in dep_ids:
            result = self.results.get(dep_id)
            if result and result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED):
                text = result.output
                if len(text) > limit:
                    logger.warning(
                        f"Context truncated for {dep_id}: "
                        f"{len(text)} → {limit} chars. "
                        f"Increase orchestrator.context_truncation_limit to avoid information loss."
                    )
                    # Keep head + tail instead of hard-cutting the end.
                    # This preserves import block at the top AND the tail
                    # (often __main__ guards, class definitions, or conclusions).
                    # Guard: if limit is very small the marker alone may exceed it;
                    # in that case fall back to a simple head-only truncation.
                    head_size = int(limit * 0.6)
                    tail_size = max(0, limit - head_size - 80)  # 80 chars for marker
                    if tail_size > 0:
                        text = (
                            text[:head_size]
                            + f"\n\n... [TRUNCATED {len(text) - limit} chars] ...\n\n"
                            + text[-tail_size:]
                        )
                    else:
                        text = text[:limit]
                parts.append(f"[Output from {dep_id}]:\n{text}")
        return "\n\n".join(parts) if parts else ""

    # ─────────────────────────────────────────
    # Status & resume
    # ─────────────────────────────────────────

    def _determine_final_status(self, state: ProjectState) -> ProjectStatus:
        # Check budget / time first — these override empty-results SYSTEM_FAILURE
        # so that a run halted by budget exhaustion or timeout is correctly labelled.
        budget_exhausted = state.budget.remaining_usd <= 0
        time_ok = state.budget.time_remaining()

        if budget_exhausted:
            return ProjectStatus.BUDGET_EXHAUSTED
        if not time_ok:
            return ProjectStatus.TIMEOUT

        if not state.results:
            return ProjectStatus.SYSTEM_FAILURE

        # COMPLETED or DEGRADED both count as "passed" for final status
        all_passed = all(
            r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)
            for r in state.results.values()
        )

        degraded_heavy = any(
            r.degraded_fallback_count > r.iterations * 0.5
            for r in state.results.values()
            if r.iterations > 0
        )

        det_ok = all(r.deterministic_check_passed for r in state.results.values())

        # Guard: ensure we actually executed ALL tasks before considering any terminal status.
        # state.results is sparse; early termination (budget exhausted, timeout) leaves tasks unexecuted.
        # If we don't check this, partial execution could be incorrectly labeled as terminal,
        # causing next run to skip _resume_project and leave unfinished tasks permanently unexecuted.
        all_tasks_executed = len(state.results) == len(state.tasks)

        if all_tasks_executed and all_passed and det_ok and not degraded_heavy:
            # All tasks executed, all passed execution, all passed validation, no degraded flag.
            # This is terminal and successful.
            return ProjectStatus.SUCCESS
        elif all_tasks_executed and all_passed and not det_ok:
            # All tasks executed and completed, but some failed deterministic validation.
            # This is a terminal status (not resumable) — completed with degraded quality.
            return ProjectStatus.COMPLETED_DEGRADED
        else:
            # Some tasks never executed (missing results), or degraded_heavy flag set, or partial execution.
            # This is resumable (genuinely incomplete) regardless of validation results on executed tasks.
            return ProjectStatus.PARTIAL_SUCCESS

    async def _resume_project(self, state: ProjectState) -> ProjectState:
        """
        Resume from last checkpoint.
        FIX #7: Restore persisted budget (spent_usd, phase_spent) instead of
        creating a fresh Budget. Only reset start_time for the new session.
        """
        # Restore budget state from checkpoint
        self.budget.spent_usd = state.budget.spent_usd
        self.budget.phase_spent = dict(state.budget.phase_spent)
        # Reset start_time so the new session gets fresh wall-clock tracking
        self.budget.start_time = time.time()

        logger.info(
            f"Restored budget: ${self.budget.spent_usd:.4f} already spent, "
            f"${self.budget.remaining_usd:.4f} remaining"
        )

        self.results = dict(state.results)
        remaining = [
            tid for tid in state.execution_order
            if tid not in self.results or
            self.results[tid].status in (TaskStatus.PENDING, TaskStatus.FAILED)
        ]
        if remaining:
            logger.info(f"Resuming: {len(remaining)} tasks remaining")
            for task_id in remaining:
                if task_id in state.tasks:
                    result = await self._execute_task(state.tasks[task_id])
                    self.results[task_id] = result
                    state.results[task_id] = result

        state.status = self._determine_final_status(state)
        return state

    def _make_state(self, project_desc: str, criteria: str,
                     tasks: dict[str, Task],
                     status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS,
                     execution_order: Optional[list[str]] = None,
                     ) -> ProjectState:
        return ProjectState(
            project_description=project_desc,
            success_criteria=criteria,
            budget=self.budget,
            tasks=tasks,
            results=dict(self.results),
            api_health={m.value: h for m, h in self.api_health.items()},
            status=status,
            execution_order=execution_order if execution_order is not None else list(tasks.keys()),
        )

    def _log_summary(self, state: ProjectState):
        logger.info("=" * 60)
        logger.info(f"PROJECT STATUS: {state.status.value}")
        logger.info(f"Budget: ${self.budget.spent_usd:.4f} / ${self.budget.max_usd}")
        logger.info(f"Time: {self.budget.elapsed_seconds:.1f}s / {self.budget.max_time_seconds}s")
        for tid, result in state.results.items():
            logger.info(
                f"  {tid}: score={result.score:.3f} status={result.status.value} "
                f"model={result.model_used.value} iters={result.iterations} "
                f"cost=${result.cost_usd:.4f}"
            )
        logger.info("=" * 60)


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
                project_path=output_dir,
                project_id=state.project_id,
                run_quality_gate=True
            )
            
            # Print summary
            summary = analyzer.generate_summary(report)
            logger.info("\n" + summary)
            
            # Save report to file
            report_file = output_dir / "analysis_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            logger.info(f"📊 Analysis report saved to: {report_file}")
            
            # Print actionable suggestions
            if report.suggestions:
                print("\n" + "="*70)
                print("💡 IMPROVEMENT SUGGESTIONS")
                print("="*70)
                
                for suggestion in report.suggestions[:5]:  # Top 5
                    priority_icon = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🔵"
                    }.get(suggestion.priority.value, "⚪")
                    
                    print(f"\n{priority_icon} [{suggestion.priority.value.upper()}] {suggestion.title}")
                    print(f"   Category: {suggestion.category.value}")
                    print(f"   Effort: {suggestion.estimated_effort}")
                    print(f"   Impact: {suggestion.expected_impact}")
                    print(f"   {suggestion.description[:100]}...")
                    
                    if suggestion.code_example:
                        print(f"\n   Example:")
                        for line in suggestion.code_example.strip().split('\n')[:3]:
                            print(f"     {line}")
                
                print("\n" + "="*70)
                print(f"💾 {len(report.suggestions)} suggestions stored in Knowledge Base")
                print("="*70)
            
        except Exception as e:
            logger.warning(f"Project analysis failed: {e}")
            # Don't fail the project if analysis fails


    async def _generate_architecture_rules(
        self,
        project_description: str,
        success_criteria: str,
        output_dir: Optional[Path]
    ) -> Optional["ProjectRules"]:
        """
        Generate architecture rules at project start.
        
        Creates .orchestrator-rules.yml with:
        - Architecture decisions
        - Technology stack
        - Constraints and patterns
        - Quality gates
        """
        try:
            from .architecture_rules import (
                ArchitectureRulesEngine,
                ProjectRules,
            )
            
            logger.info("🏗️ Generating architecture rules...")
            
            engine = ArchitectureRulesEngine(client=self.client)
            rules = await engine.generate_rules(
                description=project_description,
                criteria=success_criteria,
            )
            
            # Print summary (decision method detected from metadata)
            summary = engine.generate_summary(rules)
            print("\n" + summary)
            
            # Save to file if output_dir provided
            if output_dir:
                rules_file = engine.save_rules(rules, output_dir)
                logger.info(f"📋 Architecture rules saved to: {rules_file}")
                
                # Also save a human-readable summary
                summary_file = output_dir / "ARCHITECTURE.md"
                summary_content = f"""# Architecture Decision

## Project Overview
- **Type**: {rules.project_type}
- **Generated**: {rules.created_at}
- **Rules Version**: {rules.version}

## Decisions

### Architecture Style
**{rules.architecture.style.value.replace('_', ' ').title()}**

{rules.architecture.rationale}

### Programming Paradigm
{rules.architecture.paradigm.value.replace('_', ' ').title()}

### Technology Stack

**Primary Language**: {rules.architecture.stack.primary_language}

**Frameworks**:
{chr(10).join(['- ' + f for f in rules.architecture.stack.frameworks])}

**Libraries**:
{chr(10).join(['- ' + l for l in rules.architecture.stack.libraries[:5]])}

**Databases**:
{chr(10).join(['- ' + d for d in rules.architecture.stack.databases])}

## Constraints

{chr(10).join(['- ' + c for c in rules.architecture.constraints])}

## Recommended Patterns

{chr(10).join(['- ' + p for p in rules.architecture.patterns])}

## Quality Gates

- **Test Coverage**: Minimum {rules.coding_standards.test_coverage_min}%
- **Max Complexity**: {rules.coding_standards.max_complexity}
- **Max Line Length**: {rules.coding_standards.max_line_length} characters
- **Type Hints**: {'Required' if rules.coding_standards.type_hints else 'Optional'}

## Tradeoffs

{chr(10).join(['- ' + t for t in rules.architecture.tradeoffs])}

---
*Generated by Multi-LLM Orchestrator Architecture Rules Engine*
"""
                summary_file.write_text(summary_content, encoding='utf-8')
                logger.info(f"📖 Architecture summary saved to: {summary_file}")
            
            return rules
            
        except Exception as e:
            logger.warning(f"Architecture rules generation failed: {e}")
            return None
