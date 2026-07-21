"""Re-export shim — canonical source: orchestrator.quality.verification"""

from orchestrator.quality.verification import *  # noqa: F401, F403

import warnings

warnings.warn(
    "verification is a deprecated re-export shim — import from orchestrator.quality.verification directly",
    DeprecationWarning,
    stacklevel=2,
)
