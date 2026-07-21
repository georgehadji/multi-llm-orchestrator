"""Re-export shim — canonical source: orchestrator.safety.accountability"""

from orchestrator.safety.accountability import *  # noqa: F401, F403

import warnings

warnings.warn(
    "accountabilit is a deprecated re-export shim — import from orchestrator.safety.accountability directly",
    DeprecationWarning,
    stacklevel=2,
)
