"""Re-export shim — canonical source: orchestrator.safety.dependency_scanner"""

from orchestrator.safety.dependency_scanner import *  # noqa: F401, F403

import warnings

warnings.warn(
    "dependency_scanner is a deprecated re-export shim — import from orchestrator.safety.dependency_scanner directly",
    DeprecationWarning,
    stacklevel=2,
)
