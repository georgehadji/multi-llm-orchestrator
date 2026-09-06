"""
performance — Backward-compatibility shim
The canonical implementation lives in orchestrator/performance.py.
New code should import from `orchestrator.performance` directly.

This copy was dead (zero live callers of any of its classes — hunt T14;
real callers of `cached`/`LRUCache` all import the root module) and had
independently hardened `QueryOptimizer.build_selective_query()` against
SQL injection (an `_ALLOWED_TABLES` allowlist plus identifier validation)
without the fix ever being backported to the root copy, which remained
vulnerable to unvalidated table/column/order_by interpolation. The
hardening is now applied at the canonical source instead.
"""

from ..performance import *  # noqa: F401, F403
