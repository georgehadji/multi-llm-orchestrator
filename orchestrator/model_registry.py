"""
Model Registry — Backward-compatibility shim
===============================================
The canonical ModelRegistry now lives in orchestrator/domain/model_registry.py.
"""

from .domain.model_registry import *  # noqa: F401, F403
