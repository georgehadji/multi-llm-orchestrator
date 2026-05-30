"""
telemetry_store — Backward-compatibility shim
The canonical implementation lives in orchestrator/state_mgmt/telemetry_store.py.
New code should import from `orchestrator.state_mgmt.telemetry_store` directly.
"""

from .state_mgmt.telemetry_store import *  # noqa: F401, F403
