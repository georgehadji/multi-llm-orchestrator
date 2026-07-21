"""Re-export shim — canonical source: orchestrator.events.ab_testing"""

from orchestrator.events.ab_testing import *  # noqa: F401, F403

import warnings

warnings.warn(
    "ab_testing is a deprecated re-export shim — import from orchestrator.events.ab_testing directly",
    DeprecationWarning,
    stacklevel=2,
)
