"""Re-export shim — canonical source: orchestrator.generators.copy_generator"""

from orchestrator.generators.copy_generator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "copy_generator is a deprecated re-export shim — import from orchestrator.generators.copy_generator directly",
    DeprecationWarning,
    stacklevel=2,
)
