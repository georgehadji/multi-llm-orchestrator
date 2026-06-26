"""
secrets_manager — Backward-compatibility shim
The canonical implementation lives in orchestrator/generators/secrets_manager.py.
New code should import from `orchestrator.generators.secrets_manager` directly.
"""

from .generators.secrets_manager import *  # noqa: F401, F403
