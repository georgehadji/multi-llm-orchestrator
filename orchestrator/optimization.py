"""Re-export shim — canonical source: orchestrator.operations.optimization"""

from orchestrator.operations.optimization import *  # noqa: F401, F403

import warnings

warnings.warn(
    "optimization is a deprecated re-export shim — import from orchestrator.operations.optimization directly",
    DeprecationWarning,
    stacklevel=2,
)
