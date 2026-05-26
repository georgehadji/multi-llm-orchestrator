"""
Multi-LLM Orchestrator v6.0 — Optimized Paradigm
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

MAJOR CHANGES v6.0:
- Dashboard consolidation: 7 dashboards → 1 core + plugins
- Event unification: 4 event systems → 1 unified bus
- Plugin extraction: Core-only + optional plugins

Quick Start:
    from orchestrator import Orchestrator, Budget

    # New unified dashboard
    from orchestrator import run_dashboard
    run_dashboard(view="mission-control")

    # New unified events
    from orchestrator import get_event_bus, ProjectStartedEvent
"""

__version__ = "6.0.0"

# ── Static imports (no circular deps — Phase 1 broke models.py ↔ budget.py) ──
# fmt: off
from .agent_model_registry import (
    AGENT_MODELS,
    AgentModelEntry,
    build_all_model_preferences,
    get_default_model_preferences,
    get_model_for,
)
from .api_clients import APIResponse, UnifiedClient
from .budget import Budget
from .cache import DiskCache
from .codebase_analyzer import CodebaseAnalyzer
from .dry_run import DryRunRenderer, ExecutionPlan, TaskPlan
from .engine import Orchestrator
from .models import (
    COST_TABLE,
    FALLBACK_CHAIN,
    Model,
    ProjectState,
    ProjectStatus,
    ROUTING_TABLE,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)
from .progress_writer import ProgressEntry, ProgressWriter
from .state import StateManager
# fmt: on
