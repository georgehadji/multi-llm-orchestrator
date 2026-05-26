"""
State — Backward-compatibility shim
======================================
The canonical StateManager now lives in orchestrator/infrastructure/state.py.
"""

from .infrastructure.state import *  # noqa: F401, F403
