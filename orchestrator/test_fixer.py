"""Re-export shim — canonical source: orchestrator.testing.fixer"""

from orchestrator.testing.fixer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "test_fixer is a deprecated re-export shim — import from orchestrator.testing.fixer directly",
    DeprecationWarning,
    stacklevel=2,
)
