"""Re-export shim — canonical source: orchestrator.generators.opengraph_generator"""

from orchestrator.generators.opengraph_generator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "opengraph_generator is a deprecated re-export shim — import from orchestrator.generators.opengraph_generator directly",
    DeprecationWarning,
    stacklevel=2,
)
