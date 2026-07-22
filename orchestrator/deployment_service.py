"""Re-export shim — canonical source: orchestrator.operations.deployment_service"""

from orchestrator.operations.deployment_service import *  # noqa: F401, F403

import warnings

warnings.warn(
    "deployment_service is a deprecated re-export shim — import from orchestrator.operations.deployment_service directly",
    DeprecationWarning,
    stacklevel=2,
)
