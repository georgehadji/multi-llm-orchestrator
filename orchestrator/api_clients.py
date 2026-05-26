"""
API Clients — Backward-compatibility shim
============================================
The canonical UnifiedClient now lives in orchestrator/infrastructure/llm_client.py.
"""

from .infrastructure.llm_client import *  # noqa: F401, F403
