"""
config_sync — Backward-compatibility shim
The canonical implementation lives in orchestrator/integrations/config_sync.py.
New code should import from `orchestrator.integrations.config_sync` directly.
"""

from .integrations.config_sync import *  # noqa: F401, F403
