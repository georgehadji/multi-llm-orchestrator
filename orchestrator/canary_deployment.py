"""
canary_deployment — Backward-compatibility shim
The canonical implementation lives in orchestrator/operations/canary_deployment.py.
New code should import from `orchestrator.operations.canary_deployment` directly.
"""

from .operations.canary_deployment import *  # noqa: F401, F403
