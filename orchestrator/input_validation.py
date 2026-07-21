"""Re-export shim — canonical source: orchestrator.safety.input_validation"""

from orchestrator.safety.input_validation import *  # noqa: F401, F403

import warnings

warnings.warn(
    "input_validation is a deprecated re-export shim — import from orchestrator.safety.input_validation directly",
    DeprecationWarning,
    stacklevel=2,
)
