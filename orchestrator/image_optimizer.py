"""Re-export shim — canonical source: orchestrator.generators.image_optimizer"""

from orchestrator.generators.image_optimizer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "image_optimizer is a deprecated re-export shim — import from orchestrator.generators.image_optimizer directly",
    DeprecationWarning,
    stacklevel=2,
)
