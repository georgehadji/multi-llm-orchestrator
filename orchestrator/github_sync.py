"""
github_sync — Backward-compatibility shim
The canonical implementation lives in orchestrator/integrations/github_sync.py.
New code should import from `orchestrator.integrations.github_sync` directly.
"""

from .integrations.github_sync import *  # noqa: F401, F403
