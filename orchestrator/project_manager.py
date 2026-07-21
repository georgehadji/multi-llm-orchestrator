"""Re-export shim — canonical source: orchestrator.project_mgmt.manager"""

from orchestrator.project_mgmt.manager import *  # noqa: F401, F403

import warnings

warnings.warn(
    "project_manager is a deprecated re-export shim — import from orchestrator.project_mgmt.manager directly",
    DeprecationWarning,
    stacklevel=2,
)
