"""Re-export shim — canonical source: orchestrator.quality.code_validator"""

from orchestrator.quality.code_validator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "code_validator is a deprecated re-export shim — import from orchestrator.quality.code_validator directly",
    DeprecationWarning,
    stacklevel=2,
)
