"""Re-export shim — canonical source: orchestrator.nash.events"""

from orchestrator.nash.events import *  # noqa: F401, F403

import warnings

warnings.warn(
    "nash_events is a deprecated re-export shim — import from orchestrator.nash.events directly",
    DeprecationWarning,
    stacklevel=2,
)
