"""
transfer_learning — Backward-compatibility shim
The canonical implementation lives in orchestrator/transfer_learning.py.
New code should import from `orchestrator.transfer_learning` directly.

This copy was dead (zero live callers — hunt T14; all real callers import
the root module) and had a stripped import justified by a stale, incorrect
comment ("meta_orchestrator types removed (module does not exist)") —
orchestrator/meta_orchestrator.py exists and defines all four names root
still imports. The stripped import made `_create_routing_proposal()` and
`_create_budget_proposal()` raise NameError on `StrategyType` the moment
they actually ran.
"""

from ..transfer_learning import *  # noqa: F401, F403
