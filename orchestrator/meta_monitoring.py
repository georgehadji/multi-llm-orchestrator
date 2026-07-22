"""Re-export shim — canonical source: orchestrator.meta.monitoring"""

from orchestrator.meta.monitoring import *  # noqa: F401, F403

import warnings

warnings.warn(
    "meta_monitoring is a deprecated re-export shim — import from orchestrator.meta.monitoring directly",
    DeprecationWarning,
    stacklevel=2,
)
