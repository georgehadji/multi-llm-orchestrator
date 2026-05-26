"""
ServiceCollection — Grouped subsystem builders for Orchestrator.
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the ENGINE_OPTIMIZATION_PLAN Phase 5: groups constructor
subsystem initialization by concern so __init__ is ~100 lines instead
of ~300.

Each slice is independently revertible via `git revert`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .accountability import AccountabilityTracker
    from .agent_safety import AgentSafetyMonitor
    from .persona import PersonaManager
    from .preflight import PreflightValidator
    from .red_team import RedTeamFramework
    from .session_watcher import SessionWatcher
    from .task_verifier import TaskVerifier
    from .token_optimizer import TokenOptimizer
    from .tool_guardrails import ToolCallGuardrailController


# ─────────────────────────────────────────────────────────────────────
# Slice A: Safety — unconditionally constructed, zero risk
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SafetyServices:
    """Safety & accountability subsystems (arXiv:2602.20021)."""

    task_verifier: TaskVerifier
    accountability: AccountabilityTracker
    agent_safety: AgentSafetyMonitor
    red_team: RedTeamFramework
    tool_guardrails: ToolCallGuardrailController

    @classmethod
    def build(cls) -> "SafetyServices":
        from .accountability import AccountabilityTracker
        from .agent_safety import AgentSafetyMonitor
        from .red_team import RedTeamFramework
        from .task_verifier import TaskVerifier
        from .tool_guardrails import ToolCallGuardrailController

        return cls(
            task_verifier=TaskVerifier(),
            accountability=AccountabilityTracker(),
            agent_safety=AgentSafetyMonitor(),
            red_team=RedTeamFramework(),
            tool_guardrails=ToolCallGuardrailController(),
        )


# ─────────────────────────────────────────────────────────────────────
# Slice B: Integration — unconditionally constructed, zero risk
# ─────────────────────────────────────────────────────────────────────


@dataclass
class IntegrationServices:
    """External projects integration (RTK, Mnemo Cortex, LiteLLM)."""

    token_optimizer: TokenOptimizer
    preflight_validator: PreflightValidator
    session_watcher: SessionWatcher
    persona_manager: PersonaManager

    @classmethod
    def build(cls) -> "IntegrationServices":
        from .persona import PersonaManager
        from .preflight import PreflightValidator
        from .session_watcher import SessionWatcher
        from .token_optimizer import TokenOptimizer

        return cls(
            token_optimizer=TokenOptimizer(),
            preflight_validator=PreflightValidator(),
            session_watcher=SessionWatcher(),
            persona_manager=PersonaManager(),
        )


# ─────────────────────────────────────────────────────────────────────
# Slice C: Search — low risk, imported unconditionally
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SearchServices:
    """Search & lifecycle infrastructure."""

    memory_manager: object  # MemoryTierManager
    bm25_search: object  # BM25Search
    reranker: object  # LLMReranker
    knowledge_base: object  # KnowledgeBase
    hybrid_pipeline: object  # HybridSearchPipeline
    rate_limiter: object  # RateLimiter
    lifecycle_manager: object  # SessionLifecycleManager

    @classmethod
    def build(cls, memory_manager: object | None = None) -> "SearchServices":
        from .bm25_search import get_bm25_search
        from .hybrid_search_pipeline import HybridSearchPipeline
        from .knowledge_base import get_knowledge_base
        from .memory_tier import MemoryTierManager
        from .query_expander import QueryExpander
        from .rate_limiter import RateLimiter
        from .reranker import get_reranker
        from .session_lifecycle import SessionLifecycleManager

        mm = memory_manager or MemoryTierManager(enable_bm25=True)
        bm25 = get_bm25_search(str(mm.storage_path / "search.db"))
        reranker = get_reranker()
        kb = get_knowledge_base()
        hybrid = HybridSearchPipeline(
            bm25_search=bm25,
            knowledge_base=kb,
            reranker=reranker,
            query_expander=QueryExpander(),
        )
        rl = RateLimiter()
        lc = SessionLifecycleManager(memory_tier_manager=mm)

        return cls(
            memory_manager=mm,
            bm25_search=bm25,
            reranker=reranker,
            knowledge_base=kb,
            hybrid_pipeline=hybrid,
            rate_limiter=rl,
            lifecycle_manager=lc,
        )


# ─────────────────────────────────────────────────────────────────────
# Slice D: Learning — medium risk, behind try/except ImportError guards
# ─────────────────────────────────────────────────────────────────────


@dataclass
class LearningServices:
    """Pattern learning, telemetry, and batch execution."""

    telemetry_store: object | None
    memory_provider_mgr: object | None
    pattern_store: object | None
    pattern_extractor: object | None
    pattern_injector: object | None
    pattern_curator: object | None
    context_compressor: object | None
    batch_runner: object | None
    batch_guard: object | None

    @classmethod
    def build(
        cls,
        telemetry_store: object | None = None,
        client: object | None = None,
        flags: object | None = None,
        task_guard_cls: type | None = None,
    ) -> "LearningServices":
        """Build with graceful fallback — any ImportError sets that field to None."""
        _ts = _ts_orig = telemetry_store
        _mm = _ps = _pe = _pi = _pc = _cc = _br = None
        _bg = None

        try:
            from .context_compressor import ContextCompressor
            from .delegation.batch_runner import BatchRunner
            from .memory.memory_manager import MemoryManager
            from .pattern_learner.curator import PatternCurator
            from .pattern_learner.extractor import PatternExtractor
            from .pattern_learner.injector import PatternInjector
            from .pattern_learner.pattern_store import PatternStore
            from .telemetry_store import TelemetryStore

            if _ts_orig is None:
                try:
                    _ts = TelemetryStore()
                except Exception:
                    _ts = None

            _mm = MemoryManager() if MemoryManager is not None else None
            _ps = PatternStore() if PatternStore is not None else None
            _pe = PatternExtractor() if PatternExtractor is not None else None
            _pi = (
                PatternInjector(store=_ps, enabled=flags.pattern_injection)
                if PatternInjector is not None and _ps is not None
                else None
            )
            _pc = (
                PatternCurator(store=_ps, client=client)
                if PatternCurator is not None and _ps is not None and client is not None
                else None
            )
            _cc = (
                ContextCompressor(client=client, enabled=flags.context_compression)
                if ContextCompressor is not None and client is not None
                else None
            )
            if flags.batch_parallelism and BatchRunner is not None:
                _br = BatchRunner(max_concurrent=flags.batch_concurrency)
            if task_guard_cls is not None:
                _bg = task_guard_cls(name="batch", max_concurrent=flags.batch_concurrency)
        except (ImportError, TimeoutError):
            pass

        return cls(
            telemetry_store=_ts,
            memory_provider_mgr=_mm,
            pattern_store=_ps,
            pattern_extractor=_pe,
            pattern_injector=_pi,
            pattern_curator=_pc,
            context_compressor=_cc,
            batch_runner=_br,
            batch_guard=_bg,
        )
