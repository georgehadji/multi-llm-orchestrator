"""
Decomposer — Backward-compatibility shim.
=============================================
The canonical implementation moved to orchestrator/application/decomposer.py.

This module re-exports Decomposer from the application layer, maintaining the
import path for existing callers.
"""

from orchestrator.application.decomposer import Decomposer  # noqa: F401
