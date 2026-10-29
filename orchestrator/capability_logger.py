"""
capability_logger — Backward-compatibility shim
The canonical implementation lives in orchestrator/state_mgmt/capability_logger.py.
New code should import from `orchestrator.state_mgmt.capability_logger` directly.
"""

from .state_mgmt.capability_logger import *  # noqa: F401, F403
