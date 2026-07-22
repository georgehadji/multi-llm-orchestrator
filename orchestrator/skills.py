"""Re-export shim — canonical source: orchestrator.skills.skills"""

from orchestrator.skills.skills import *  # noqa: F401, F403

import warnings

warnings.warn(
    "skills is a deprecated re-export shim — import from orchestrator.skills.skills directly",
    DeprecationWarning,
    stacklevel=2,
)
