"""
tenancy — Backward-compatibility shim
The canonical implementation lives in orchestrator/integrations/tenancy.py.
New code should import from `orchestrator.integrations.tenancy` directly.
"""

from .integrations.tenancy import *  # noqa: F401, F403
