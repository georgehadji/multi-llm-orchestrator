"""
Task Factory — Backward-compatibility shim
============================================
The canonical TaskFactory now lives in orchestrator/domain/task_factory.py.
"""

from .domain.task_factory import *  # noqa: F401, F403
