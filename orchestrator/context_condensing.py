"""Re-export shim — canonical source: orchestrator.context_mgmt.condensing"""

from orchestrator.context_mgmt.condensing import *  # noqa: F401, F403

import warnings

warnings.warn(
    "context_condensing is a deprecated re-export shim — import from orchestrator.context_mgmt.condensing directly",
    DeprecationWarning,
    stacklevel=2,
)
