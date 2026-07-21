"""Re-export shim — canonical source: orchestrator.generators.logging_generator"""

from orchestrator.generators.logging_generator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "logging_generator is a deprecated re-export shim — import from orchestrator.generators.logging_generator directly",
    DeprecationWarning,
    stacklevel=2,
)
