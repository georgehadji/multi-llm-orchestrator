"""Re-export shim — canonical source: orchestrator.nash.stable_orchestrator"""

from orchestrator.nash.stable_orchestrator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "nash_stable_orchestrator is a deprecated re-export shim — import from orchestrator.nash.stable_orchestrator directly",
    DeprecationWarning,
    stacklevel=2,
)
