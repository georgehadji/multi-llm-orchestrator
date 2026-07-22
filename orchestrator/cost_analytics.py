"""Re-export shim — canonical source: orchestrator.costing.analytics"""

from orchestrator.costing.analytics import *  # noqa: F401, F403

import warnings

warnings.warn(
    "cost_analytics is a deprecated re-export shim — import from orchestrator.costing.analytics directly",
    DeprecationWarning,
    stacklevel=2,
)
