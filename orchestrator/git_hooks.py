"""Re-export shim — canonical source: orchestrator.vcs.hooks"""

from orchestrator.vcs.hooks import *  # noqa: F401, F403

import warnings

warnings.warn(
    "git_hooks is a deprecated re-export shim — import from orchestrator.vcs.hooks directly",
    DeprecationWarning,
    stacklevel=2,
)
