"""
tdd_config — Backward-compatibility shim
The canonical implementation lives in orchestrator/quality/tdd_config.py.
New code should import from `orchestrator.quality.tdd_config` directly.
"""

from .quality.tdd_config import *  # noqa: F401, F403
