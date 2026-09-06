"""
federated_learning — Backward-compatibility shim
The canonical implementation lives in orchestrator/federated_learning.py.
New code should import from `orchestrator.federated_learning` directly.

This copy was dead (zero live callers — hunt T14; the only real caller,
nash/stable_orchestrator.py, imports the root module) and had a stripped
import (`# REMOVED: from .feedback_loop import CodebaseFingerprint,
OutcomeStatus, ProductionOutcome`) that made `contribute_insight()` raise
NameError on `OutcomeStatus` the moment it actually ran.
"""

from ..federated_learning import *  # noqa: F401, F403
