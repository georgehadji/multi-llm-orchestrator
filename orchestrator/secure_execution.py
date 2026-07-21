"""Re-export shim — canonical source: orchestrator.safety.secure_execution"""

from orchestrator.safety.secure_execution import *  # noqa: F401, F403

import warnings

warnings.warn(
    "secure_execution is a deprecated re-export shim — import from orchestrator.safety.secure_execution directly",
    DeprecationWarning,
    stacklevel=2,
)
