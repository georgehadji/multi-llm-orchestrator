"""Re-export shim — canonical source: orchestrator.context_mgmt.sources"""

from orchestrator.context_mgmt.sources import *  # noqa: F401, F403

import warnings

warnings.warn(
    "context_sources is a deprecated re-export shim — import from orchestrator.context_mgmt.sources directly",
    DeprecationWarning,
    stacklevel=2,
)
