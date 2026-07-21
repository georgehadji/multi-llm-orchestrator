"""Re-export shim — canonical source: orchestrator.infrastructure.streaming_resilient"""

from orchestrator.infrastructure.streaming_resilient import *  # noqa: F401, F403

import warnings

warnings.warn(
    "streaming_resilient is a deprecated re-export shim — import from orchestrator.infrastructure.streaming_resilient directly",
    DeprecationWarning,
    stacklevel=2,
)
