"""
Ports — Backward-compatibility shim
=====================================
The canonical port definitions now live in orchestrator/domain/ports.py.
This file is a backward-compatibility re-export.
"""

from .domain.ports import *  # noqa: F401, F403
