"""Re-export shim — canonical source: orchestrator.testing.validator"""

from orchestrator.testing.validator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "test_validator is a deprecated re-export shim — import from orchestrator.testing.validator directly",
    DeprecationWarning,
    stacklevel=2,
)
