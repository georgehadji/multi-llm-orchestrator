"""Re-export shim — canonical source: orchestrator.context_mgmt.compressor"""

from orchestrator.context_mgmt.compressor import *  # noqa: F401, F403

import warnings

warnings.warn(
    "context_compressor is a deprecated re-export shim — import from orchestrator.context_mgmt.compressor directly",
    DeprecationWarning,
    stacklevel=2,
)
