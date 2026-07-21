"""Re-export shim — canonical source: orchestrator.operations.drift"""

from orchestrator.operations.drift import *  # noqa: F401, F403

import warnings

warnings.warn(
    "drift is a deprecated re-export shim — import from orchestrator.operations.drift directly",
    DeprecationWarning,
    stacklevel=2,
)
