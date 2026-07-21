"""Re-export shim — canonical source: orchestrator.project_mgmt.file"""

from orchestrator.project_mgmt.file import *  # noqa: F401, F403

import warnings

warnings.warn(
    "project_file is a deprecated re-export shim — import from orchestrator.project_mgmt.file directly",
    DeprecationWarning,
    stacklevel=2,
)
