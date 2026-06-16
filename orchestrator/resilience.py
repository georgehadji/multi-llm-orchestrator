"""
Resilience — Backward-compatibility shim.
===========================================
The canonical implementation is in orchestrator/operations/resilience.py.
Import from there directly for new code; this module exists only for callers
that have not yet been migrated.
"""

from orchestrator.operations.resilience import (  # noqa: F401
    CascadePolicy,
    CostTier,
    FallbackTriggeredEvent,
    ResiliencePolicy,
    RetryTemplate,
    classify_model_tier,
    resolve_fallback_chain,
    run_with_resilience,
)
