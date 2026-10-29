"""
code_executor — Backward-compatibility shim
The canonical implementation lives in orchestrator/safety/code_executor.py.
New code should import from `orchestrator.safety.code_executor` directly.
"""

from .safety.code_executor import *  # noqa: F401, F403
