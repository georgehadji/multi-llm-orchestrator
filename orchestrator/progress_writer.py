"""
progress_writer — Backward-compatibility shim
The canonical implementation lives in orchestrator/analysis/progress_writer.py.
New code should import from `orchestrator.analysis.progress_writer` directly.
"""

from .analysis.progress_writer import *  # noqa: F401, F403
