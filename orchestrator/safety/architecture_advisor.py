"""
architecture_advisor — Backward-compatibility shim
The canonical implementation lives in orchestrator/architecture_advisor.py.
New code should import from `orchestrator.architecture_advisor` directly.

This copy was dead (zero live callers, confirmed by exhaustive repo-wide
grep — hunt T9), carried a broken relative import
(`from ...api_clients import UnifiedClient`, one dot too many), and had
silently fallen behind the canonical module (missing the "static" project
type's tech-stack/topology entries the canonical version has).
"""

from ..architecture_advisor import *  # noqa: F401, F403
