"""
metrics — Backward-compatibility shim
The canonical implementation lives in orchestrator/analysis/metrics.py.
New code should import from `orchestrator.analysis.metrics` directly.
"""

from .analysis.metrics import *  # noqa: F401, F403
