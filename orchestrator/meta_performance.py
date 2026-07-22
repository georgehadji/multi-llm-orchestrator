"""Re-export shim — canonical source: orchestrator.meta.performance"""

from orchestrator.meta.performance import *  # noqa: F401, F403

import warnings

warnings.warn(
    "meta_performance is a deprecated re-export shim — import from orchestrator.meta.performance directly",
    DeprecationWarning,
    stacklevel=2,
)
