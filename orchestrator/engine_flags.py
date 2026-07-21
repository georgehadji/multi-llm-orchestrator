"""
Engine Feature Flag Imports — consolidated import gating
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Consolidates the 20+ ``if flags.X_enabled: try: from .module import Y``
blocks that were scattered across ``Orchestrator.__init__`` into a single
declarative registry.

Usage:
    from .engine_flags import FEATURE_IMPORTS, import_feature_modules

    imported = import_feature_modules(flags)
    # imported["a2a_manager"] -> A2AManager | None
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .crosscutting.config import FeatureFlags

logger = logging.getLogger("orchestrator.engine_flags")

# Registry: flag_name -> (module_path, attr_names, default)
# Each entry is imported only when flags.flag_name is True.
FEATURE_IMPORTS: dict[str, tuple[str, list[str], Any]] = {
    "cache_optimizer_enabled": (
        "orchestrator.cache_optimizer",
        ["CacheOptimizer", "CacheConfig"],
        None,
    ),
    "test_validation_enabled": (
        "orchestrator.test_validator",
        ["TestValidator", "validate_and_generate_test"],
        None,
    ),
    "code_validation_enabled": (
        "orchestrator.code_validator",
        ["validate_code", "extract_code_from_llm_response"],
        None,
    ),
    "a2a_enabled": (
        "orchestrator.a2a_manager",
        ["A2AManager", "AgentCard"],
        None,
    ),
    "accountability_enabled": (
        "orchestrator.accountability",
        ["AccountabilityTracker", "ActionType", "ActorType"],
        None,
    ),
    "agent_safety_enabled": (
        "orchestrator.agent_safety",
        ["AgentSafetyMonitor", "SafetyEventType"],
        None,
    ),
    "audit_log": (
        "orchestrator.audit",
        ["AuditLog"],
        None,
    ),
    "bm25_search_enabled": (
        "orchestrator.bm25_search",
        ["BM25Search", "get_bm25_search"],
        None,
    ),
    "cost_optimization_enabled": (
        "orchestrator.cost_optimization",
        [
            "ModelCascader",
            "TokenBudget",
            "SpeculativeGenerator",
            "PromptCacher",
            "BatchClient",
            "AdaptiveTemperatureController",
            "StreamingValidator",
            "DependencyContextInjector",
        ],
        None,
    ),
    "memory_tier_enabled": (
        "orchestrator.memory_tier",
        ["MemoryTierManager"],
        None,
    ),
    "persona_enabled": (
        "orchestrator.persona",
        ["PersonaManager", "PersonaMode"],
        None,
    ),
    "red_team_enabled": (
        "orchestrator.red_team",
        ["RedTeamFramework"],
        None,
    ),
    "reranker_enabled": (
        "orchestrator.reranker",
        ["LLMReranker", "get_reranker"],
        None,
    ),
    "session_lifecycle_enabled": (
        "orchestrator.session_lifecycle",
        ["SessionLifecycleManager"],
        None,
    ),
    "session_watcher_enabled": (
        "orchestrator.session_watcher",
        ["SessionWatcher"],
        None,
    ),
    "task_verifier_enabled": (
        "orchestrator.task_verifier",
        ["TaskVerifier"],
        None,
    ),
    "token_optimizer_enabled": (
        "orchestrator.token_optimizer",
        ["TokenOptimizer"],
        None,
    ),
    "tracing_enabled": (
        "orchestrator.tracing",
        ["TracingConfig", "configure_tracing", "get_tracer", "traced_task"],
        None,
    ),
    "tdd_enabled": (
        "orchestrator.test_first_generator",
        ["TestFirstGenerator", "TDDResult"],
        None,
    ),
    "diff_generation_enabled": (
        "orchestrator.diff_generator",
        ["DiffGenerator", "DiffResult", "apply_unified_diff"],
        None,
    ),
}


def import_feature_modules(flags: FeatureFlags) -> dict[str, Any]:
    """
    Conditionally import optional feature modules based on flags.

    Returns a dict mapping each attribute name to its imported value (or None
    if the feature is disabled or the import fails).

    Usage in ``Orchestrator.__init__``::
        _mods = import_feature_modules(flags)
        CacheOptimizer = _mods.get("CacheOptimizer")
    """
    result: dict[str, Any] = {}

    for flag_name, (module_path, attr_names, default) in FEATURE_IMPORTS.items():
        enabled = getattr(flags, flag_name, False)
        if not enabled:
            for name in attr_names:
                result[name] = default
            continue

        try:
            mod = importlib.import_module(module_path)
            for name in attr_names:
                result[name] = getattr(mod, name, default)
        except (ImportError, AttributeError) as exc:
            logger.debug(
                "Feature %s (flag=%s) not available: %s",
                flag_name,
                enabled,
                exc,
            )
            for name in attr_names:
                result[name] = default

    return result
