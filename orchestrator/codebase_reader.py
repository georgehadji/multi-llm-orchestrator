"""Re-export shim — canonical source: orchestrator.codebase.reader"""

from orchestrator.codebase.reader import *  # noqa: F401, F403

import warnings

warnings.warn(
    "codebase_reader is a deprecated re-export shim — import from orchestrator.codebase.reader directly",
    DeprecationWarning,
    stacklevel=2,
)
