"""Re-export shim — canonical source: orchestrator.operations.error_handling"""

from orchestrator.operations.error_handling import *  # noqa: F401, F403

import warnings

warnings.warn(
    "error_handling is a deprecated re-export shim — import from orchestrator.operations.error_handling directly",
    DeprecationWarning,
    stacklevel=2,
)
