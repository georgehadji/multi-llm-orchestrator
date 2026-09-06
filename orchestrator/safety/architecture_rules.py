"""
architecture_rules — Backward-compatibility shim
The canonical implementation lives in orchestrator/architecture_rules.py.
New code should import from `orchestrator.architecture_rules` directly.

This copy was dead (zero live callers, confirmed by exhaustive repo-wide
grep — hunt T9) and carried a broken relative import
(`from ...models import Model as M`, one dot too many for this file's own
depth) that would raise ImportError the moment anything actually reached it.
"""

from ..architecture_rules import *  # noqa: F401, F403
