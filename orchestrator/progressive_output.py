"""
progressive_output — Backward-compatibility shim
The canonical implementation lives in orchestrator/analysis/progressive_output.py.
New code should import from `orchestrator.analysis.progressive_output` directly.
"""

from .analysis.progressive_output import *  # noqa: F401, F403
