"""
preflight — Backward-compatibility shim
The canonical implementation lives in orchestrator/quality/preflight.py.
New code should import from `orchestrator.quality.preflight` directly.
"""

from .quality.preflight import *  # noqa: F401, F403
