"""Re-export shim — canonical source: orchestrator.workspace.workspace"""

from orchestrator.workspace.workspace import *  # noqa: F401, F403

import warnings

warnings.warn(
    "workspace is a deprecated re-export shim — import from orchestrator.workspace.workspace directly",
    DeprecationWarning,
    stacklevel=2,
)
