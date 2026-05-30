"""
data_sources — Backward-compatibility shim
The canonical implementation lives in orchestrator/connectors/data_sources.py.
New code should import from `orchestrator.connectors.data_sources` directly.
"""

from .connectors.data_sources import *  # noqa: F401, F403
