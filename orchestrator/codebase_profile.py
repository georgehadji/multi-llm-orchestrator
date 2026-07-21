"""Re-export shim — canonical source: orchestrator.codebase.profile"""

from orchestrator.codebase.profile import *  # noqa: F401, F403

import warnings

warnings.warn(
    "codebase_profile is a deprecated re-export shim — import from orchestrator.codebase.profile directly",
    DeprecationWarning,
    stacklevel=2,
)
