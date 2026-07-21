"""Re-export shim — canonical source: orchestrator.codebase.analyzer"""

from orchestrator.codebase.analyzer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "codebase_analyzer is a deprecated re-export shim — import from orchestrator.codebase.analyzer directly",
    DeprecationWarning,
    stacklevel=2,
)
