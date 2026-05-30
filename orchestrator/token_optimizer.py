"""
TokenOptimizer — Backward-compatibility shim
=============================================
Canonical location: orchestrator/infrastructure/token_optimizer.py
Import from there directly for new code; this shim exists for existing callers.
"""

from .infrastructure.token_optimizer import (  # noqa: F401
    TokenOptimizer,
    get_global_token_optimizer,
)

__all__ = ["TokenOptimizer", "get_global_token_optimizer"]
