"""
Task Factory — Backward-compatibility shim
============================================
The canonical TaskFactory now lives in orchestrator/domain/task_factory.py.
"""

from ..domain.task_factory import TaskFactory  # noqa: F401

__all__ = ["TaskFactory"]
