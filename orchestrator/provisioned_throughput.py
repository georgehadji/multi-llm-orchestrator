"""
provisioned_throughput — Backward-compatibility shim
The canonical implementation lives in orchestrator/operations/provisioned_throughput.py
(the same module orchestrator/infrastructure/provisioned_throughput.py already
shims to). New code should import from
`orchestrator.operations.provisioned_throughput` directly.

This copy was a byte-for-byte independent duplicate (not a shim relationship)
of the canonical module — confirmed identical today via diff, but with zero
live callers of any of the three copies anywhere in the repo (hunt T10). An
unshimmed duplicate pair is exactly the shape every prior tier (T1, T2, T3,
T5, T7, T9) has found silently diverging once one copy gets a fix the other
doesn't. Converted now, before that happens.
"""

from .operations.provisioned_throughput import *  # noqa: F401, F403
