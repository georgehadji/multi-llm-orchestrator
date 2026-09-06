"""Frontend security rules — re-export shim from root.

Was a byte-identical 1058-line duplicate of
``orchestrator/frontend_security.py`` exposed through ``design/__init__.py``'s
wildcard — the largest such duplicate found in this repository (hunt T17).
"""

from ..frontend_security import *  # noqa: F401, F403
