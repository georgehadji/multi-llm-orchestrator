"""
Tracing — Backward-compatibility shim
======================================
Canonical location: orchestrator/infrastructure/tracing.py
Import from there directly for new code; this shim exists for existing callers.
"""

from .infrastructure.tracing import (  # noqa: F401
    Span,
    Tracer,
    TracingConfig,
    get_tracer,
    set_global_tracer,
    trace_function,
    traced_llm_call,
    traced_policy_check,
    traced_remediation,
    traced_task,
)

__all__ = [
    "Span",
    "Tracer",
    "TracingConfig",
    "get_tracer",
    "set_global_tracer",
    "trace_function",
    "traced_llm_call",
    "traced_policy_check",
    "traced_remediation",
    "traced_task",
]
