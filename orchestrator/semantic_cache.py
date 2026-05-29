"""
SemanticCache — Backward-compatibility shim
=============================================
Canonical location: orchestrator/infrastructure/semantic_cache.py
Import from there directly for new code; this shim exists for existing callers.
"""
from .infrastructure.semantic_cache import SemanticCache, SemanticPattern  # noqa: F401

__all__ = ["SemanticCache", "SemanticPattern"]
