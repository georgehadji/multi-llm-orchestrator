"""Re-export shim — canonical source: orchestrator.context_mgmt.dedup"""

from orchestrator.context_mgmt.dedup import *  # noqa: F401, F403

import warnings

warnings.warn(
    "context_dedu is a deprecated re-export shim — import from orchestrator.context_mgmt.dedup directly",
    DeprecationWarning,
    stacklevel=2,
)
