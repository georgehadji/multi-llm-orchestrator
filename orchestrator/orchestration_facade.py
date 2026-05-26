"""
OrchestratorFacade — Safe accessors for optional subsystems.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the ENGINE_OPTIMIZATION_PLAN Phase 1: groups 33 thin property-accessor
methods that currently live on Orchestrator.  Every method returns a reference
to an optional sub-module — they exist because the modules may not import
(try/except ImportError in engine.py) and callers need a safe access point.

Usage:
    facade = OrchestratorFacade.from_orchestrator(orch)
    a2a = facade.a2a_manager  # returns A2AManager or None
    reranker = facade.require("reranker")  # raises if not available
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .a2a_protocol import A2AManager
    from .accountability import AccountabilityTracker
    from .agent_safety import AgentSafetyMonitor
    from .bm25_search import BM25Search
    from .hybrid_search_pipeline import HybridSearchPipeline
    from .memory_tier import MemoryTierManager
    from .persona import PersonaManager
    from .preflight import PreflightValidator
    from .rate_limiter import RateLimiter
    from .red_team import RedTeamFramework
    from .reranker import LLMReranker
    from .session_lifecycle import SessionLifecycleManager
    from .session_watcher import SessionWatcher
    from .task_verifier import TaskVerifier
    from .telemetry_store import TelemetryStore
    from .token_optimizer import TokenOptimizer


@dataclass
class OrchestratorFacade:
    """Safe accessors for optional subsystems.

    Every attribute is the underlying private field from Orchestrator.
    Constructed via `from_orchestrator(orch)` which reads the
    underscore-prefixed attributes.
    """

    # ── A2A / Messaging ──
    a2a_manager: A2AManager | None = None
    channels: dict[str, Any] = field(default_factory=dict)

    # ── Search Infrastructure ──
    bm25_search: BM25Search | None = None
    reranker: LLMReranker | None = None
    hybrid_pipeline: HybridSearchPipeline | None = None
    knowledge_base: object | None = None
    query_expander: object | None = None

    # ── Persona & Session ──
    persona_manager: PersonaManager | None = None
    session_watcher: SessionWatcher | None = None
    lifecycle_manager: SessionLifecycleManager | None = None
    memory_manager: MemoryTierManager | None = None

    # ── Safety & Security ──
    preflight_validator: PreflightValidator | None = None
    token_optimizer: TokenOptimizer | None = None
    red_team: RedTeamFramework | None = None
    agent_safety: AgentSafetyMonitor | None = None
    accountability: AccountabilityTracker | None = None
    task_verifier: TaskVerifier | None = None
    tool_guardrails: object | None = None

    # ── Rate Limiting ──
    rate_limiter: RateLimiter | None = None

    # ── Telemetry & Learning ──
    telemetry_store: TelemetryStore | None = None

    @classmethod
    def from_orchestrator(cls, orch: object) -> OrchestratorFacade:
        """Build facade by reading private attributes from an Orchestrator instance."""
        return cls(
            a2a_manager=getattr(orch, "_a2a_manager", None),
            channels=getattr(orch, "_channels", {}),
            bm25_search=getattr(orch, "_bm25_search", None),
            reranker=getattr(orch, "_reranker", None),
            hybrid_pipeline=getattr(orch, "_hybrid_pipeline", None),
            knowledge_base=getattr(orch, "_knowledge_base", None),
            persona_manager=getattr(orch, "_persona_manager", None),
            session_watcher=getattr(orch, "_session_watcher", None),
            lifecycle_manager=getattr(orch, "_lifecycle_manager", None),
            memory_manager=getattr(orch, "_memory_manager", None),
            preflight_validator=getattr(orch, "_preflight_validator", None),
            token_optimizer=getattr(orch, "_token_optimizer", None),
            red_team=getattr(orch, "_red_team", None),
            agent_safety=getattr(orch, "_agent_safety", None),
            accountability=getattr(orch, "_accountability", None),
            task_verifier=getattr(orch, "_task_verifier", None),
            tool_guardrails=getattr(orch, "_tool_guardrails", None),
            rate_limiter=getattr(orch, "_rate_limiter", None),
            telemetry_store=getattr(orch, "_telemetry_store", None),
        )

    def require(self, name: str) -> object:
        """Return subsystem by name, or raise AttributeError if not available."""
        val = getattr(self, name, None)
        if val is None:
            raise AttributeError(f"Subsystem '{name}' is not available (optional module not imported)")
        return val
