"""Re-export shim — canonical source: orchestrator.costing.tracker"""

from orchestrator.costing.tracker import *  # noqa: F401, F403

import warnings

warnings.warn(
    "cost_tracker is a deprecated re-export shim — import from orchestrator.costing.tracker directly",
    DeprecationWarning,
    stacklevel=2,
)
