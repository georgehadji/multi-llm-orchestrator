"""Re-export shim — canonical source: orchestrator.nash.auto_tuning"""

from orchestrator.nash.auto_tuning import *  # noqa: F401, F403

import warnings

warnings.warn(
    "nash_auto_tuning is a deprecated re-export shim — import from orchestrator.nash.auto_tuning directly",
    DeprecationWarning,
    stacklevel=2,
)
