"""Re-export shim — canonical source: orchestrator.operations.breakpoints"""

from orchestrator.operations.breakpoints import *  # noqa: F401, F403

import warnings

warnings.warn(
    "breakpoints is a deprecated re-export shim — import from orchestrator.operations.breakpoints directly",
    DeprecationWarning,
    stacklevel=2,
)
