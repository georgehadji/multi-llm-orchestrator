"""Re-export shim — canonical source: orchestrator.routing.routing"""

from orchestrator.routing.routing import *  # noqa: F401, F403

import warnings

warnings.warn(
    "model_routing is a deprecated re-export shim — import from orchestrator.routing.routing directly",
    DeprecationWarning,
    stacklevel=2,
)
