"""Re-export shim — canonical source: orchestrator.infrastructure.streaming_optimizer"""

from orchestrator.infrastructure.streaming_optimizer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "streaming_optimizer is a deprecated re-export shim — import from orchestrator.infrastructure.streaming_optimizer directly",
    DeprecationWarning,
    stacklevel=2,
)
