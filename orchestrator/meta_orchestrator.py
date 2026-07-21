"""Re-export shim — canonical source: orchestrator.meta.orchestrator"""

from orchestrator.meta.orchestrator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "meta_orchestrator is a deprecated re-export shim — import from orchestrator.meta.orchestrator directly",
    DeprecationWarning,
    stacklevel=2,
)
