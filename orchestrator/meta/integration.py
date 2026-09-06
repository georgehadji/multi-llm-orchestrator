"""
integration — Backward-compatibility shim
The canonical implementation lives in orchestrator/meta_integration.py.
New code should import from `orchestrator.meta_integration` directly.

This copy was dead (zero live callers — hunt T12; all four real call sites
import the root module directly) and had fallen out of sync with a fix
already applied to the root copy: `_state_to_trajectory` here read
`state.status.value` directly, raising AttributeError whenever `state.status`
was a plain string rather than an enum. The root copy normalizes both shapes
via `getattr(state.status, "value", state.status)`.
"""

from ..meta_integration import *  # noqa: F401, F403
