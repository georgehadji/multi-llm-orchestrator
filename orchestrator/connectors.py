"""
connectors — Backward-compatibility shim
The canonical implementation lives in orchestrator/connectors/connectors.py.
New code should import from `orchestrator.connectors.connectors` directly.
"""

from .connectors.connectors import *  # noqa: F401, F403
