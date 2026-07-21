"""Re-export shim — canonical source: orchestrator.codebase.context"""

from orchestrator.codebase.context import *  # noqa: F401, F403

import warnings

warnings.warn(
    "codebase_context is a deprecated re-export shim — import from orchestrator.codebase.context directly",
    DeprecationWarning,
    stacklevel=2,
)
