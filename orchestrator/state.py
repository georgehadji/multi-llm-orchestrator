"""
State — Backward-compatibility shim
======================================
The canonical StateManager now lives in orchestrator/infrastructure/state.py.
"""

from .infrastructure.state import StateManager  # noqa: F401
from .infrastructure.state import DEFAULT_STATE_PATH  # noqa: F401
from .infrastructure.state import migrate_add_resume_fields  # noqa: F401
from .infrastructure.state import extract_and_store_keywords  # noqa: F401

__all__ = [
    "StateManager",
    "DEFAULT_STATE_PATH",
    "migrate_add_resume_fields",
    "extract_and_store_keywords",
]
