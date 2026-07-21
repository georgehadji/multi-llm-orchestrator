"""Re-export shim — canonical source: orchestrator.vcs.service"""

from orchestrator.vcs.service import *  # noqa: F401, F403

import warnings

warnings.warn(
    "git_service is a deprecated re-export shim — import from orchestrator.vcs.service directly",
    DeprecationWarning,
    stacklevel=2,
)
