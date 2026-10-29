"""
visualization — Backward-compatibility shim
The canonical implementation lives in orchestrator/analysis/visualization.py.
New code should import from `orchestrator.analysis.visualization` directly.
"""

from .analysis.visualization import *  # noqa: F401, F403
