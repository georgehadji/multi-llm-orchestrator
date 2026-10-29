"""
assumption_gate — Backward-compatibility shim
The canonical implementation lives in orchestrator/analysis/assumption_gate.py.
New code should import from `orchestrator.analysis.assumption_gate` directly.
"""

from .analysis.assumption_gate import *  # noqa: F401, F403
