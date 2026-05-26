"""
Budget — Re-export shim for the Budget class
=============================================
The Budget class was moved to orchestrator/domain/models.py in Phase 1
of the Architectural Remediation Plan to break a circular import loop.

This file is kept as a backward-compatibility shim. All code should
import Budget from `orchestrator.models` going forward.
"""

from .models import Budget  # noqa: F401
