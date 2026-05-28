"""
API Clients — Backward-compatibility shim
==========================================
The canonical UnifiedClient now lives in orchestrator/infrastructure/llm_client.py.
Import from there directly for new code; this module exists only for callers that
have not yet been migrated.
"""

from .infrastructure.llm_client import (
    APIResponse,
    AuthenticationError,
    UnifiedClient,
    validate_model_available,
)

__all__ = [
    "APIResponse",
    "AuthenticationError",
    "UnifiedClient",
    "validate_model_available",
]
