"""Re-export shim — canonical source: orchestrator.project_mgmt.copier"""

from orchestrator.project_mgmt.copier import *  # noqa: F401, F403

import warnings

warnings.warn(
    "project_copier is a deprecated re-export shim — import from orchestrator.project_mgmt.copier directly",
    DeprecationWarning,
    stacklevel=2,
)
