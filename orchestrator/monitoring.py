"""Re-export shim — canonical source: orchestrator.infrastructure.monitoring"""

from orchestrator.infrastructure.monitoring import *  # noqa: F401, F403

import warnings

warnings.warn(
    "monitoring is a deprecated re-export shim — import from orchestrator.infrastructure.monitoring directly",
    DeprecationWarning,
    stacklevel=2,
)
