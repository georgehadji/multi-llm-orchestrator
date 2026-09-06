"""
token_budget — Backward-compatibility shim
The canonical implementation lives in orchestrator/infrastructure/token_budget.py.
New code should import from `orchestrator.infrastructure.token_budget` directly.

This copy was a byte-for-byte independent duplicate (not a shim relationship)
of the canonical module — confirmed identical today via diff, but with zero
live callers of either copy anywhere in the repo (hunt T10), an unshimmed
duplicate pair is exactly the shape every prior tier (T1, T2, T3, T5, T7, T9)
has found silently diverging once one copy gets a fix the other doesn't.
Converted now, before that happens.

Not to be confused with orchestrator/cost_optimization/token_budget.py — a
different, unrelated class (`TokenBudget`, per-phase output-token ceilings)
despite the similar filename; see docs/hunts/t10-cost-optimization/inventory.md.
"""

from .infrastructure.token_budget import *  # noqa: F401, F403
