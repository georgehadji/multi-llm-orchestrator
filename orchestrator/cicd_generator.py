"""Re-export shim — canonical source: orchestrator.generators.cicd_generator"""

from orchestrator.generators.cicd_generator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "cicd_generator is a deprecated re-export shim — import from orchestrator.generators.cicd_generator directly",
    DeprecationWarning,
    stacklevel=2,
)
