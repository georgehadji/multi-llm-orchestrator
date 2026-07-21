"""Re-export shim — canonical source: orchestrator.context_mgmt.truncator"""

from orchestrator.context_mgmt.truncator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "context_truncator is a deprecated re-export shim — import from orchestrator.context_mgmt.truncator directly",
    DeprecationWarning,
    stacklevel=2,
)
