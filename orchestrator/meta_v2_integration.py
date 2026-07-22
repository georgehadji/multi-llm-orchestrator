"""Re-export shim — canonical source: orchestrator.meta.v2_integration"""

from orchestrator.meta.v2_integration import *  # noqa: F401, F403

import warnings

warnings.warn(
    "meta_v2_integration is a deprecated re-export shim — import from orchestrator.meta.v2_integration directly",
    DeprecationWarning,
    stacklevel=2,
)
