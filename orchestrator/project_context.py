"""Re-export shim — canonical source: orchestrator.project_mgmt.context"""

from orchestrator.project_mgmt.context import *  # noqa: F401, F403

import warnings

warnings.warn(
    "project_context is a deprecated re-export shim — import from orchestrator.project_mgmt.context directly",
    DeprecationWarning,
    stacklevel=2,
)
