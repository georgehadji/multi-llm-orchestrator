"""
Resilience — Backward-compatibility shim.
===========================================
The canonical implementation is in orchestrator/operations/resilience.py.
Import from there directly for new code; this module exists only for callers
that have not yet been migrated.
"""

from orchestrator.operations.resilience import (
    CascadePolicy,
    CostTier,
    FallbackTriggeredEvent,
    ResiliencePolicy,
    RetryTemplate,
    classify_model_tier,
    resolve_fallback_chain,
    run_with_resilience,
)

# Explicit re-export so this shim's names are importable by name (mypy treats
# imported names as non-exported unless listed in __all__ or aliased).
__all__ = [
    "CascadePolicy",
    "CostTier",
    "FallbackTriggeredEvent",
    "ResiliencePolicy",
    "RetryTemplate",
    "classify_model_tier",
    "resolve_fallback_chain",
    "run_with_resilience",
]
