"""Re-export shim — canonical source: orchestrator.safety.sandbox"""

from orchestrator.safety.sandbox import *  # noqa: F401, F403

import warnings

warnings.warn(
    "sandbox is a deprecated re-export shim — import from orchestrator.safety.sandbox directly",
    DeprecationWarning,
    stacklevel=2,
)
