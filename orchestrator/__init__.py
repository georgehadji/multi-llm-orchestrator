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


# Lazy-load everything to prevent circular imports
def __getattr__(name: str):
    """Lazy-load modules on-demand to prevent circular imports."""
    if name == "__all__":
        return []

    # Import on first access
    from importlib import import_module

    # Try to find the attribute in submodules
    common_modules = {
        "Orchestrator": "engine",
        "Budget": "budget",
        "DiskCache": "cache",
        "Model": "models",
        "ProjectState": "models",
        "Task": "models",
        "TaskResult": "models",
        "TaskType": "models",
        "TaskStatus": "models",
        "ProjectStatus": "models",
        "StateManager": "state",
        "APIResponse": "api_clients",
        "UnifiedClient": "api_clients",
        "COST_TABLE": "models",
        "ROUTING_TABLE": "models",
        "FALLBACK_CHAIN": "models",
        "CodebaseAnalyzer": "codebase_analyzer",
        "ExecutionPlan": "dry_run",
        "TaskPlan": "dry_run",
        "DryRunRenderer": "dry_run",
        "ProgressWriter": "progress_writer",
        "ProgressEntry": "progress_writer",
        "CodebaseUnderstanding": "codebase_understanding",
        "ImprovementSuggester": "improvement_suggester",
        "CodebaseProfile": "codebase_profile",
    }

    if name in common_modules:
        module_name = common_modules[name]
        module = import_module(f".{module_name}", package=__name__)
        return getattr(module, name, None)

    # Try importing the name as a module
    try:
        return import_module(f".{name}", package=__name__)
    except ImportError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
