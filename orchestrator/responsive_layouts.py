"""Re-export shim — canonical source: orchestrator.design.responsive_layouts"""

from orchestrator.design.responsive_layouts import *  # noqa: F401, F403

import warnings

warnings.warn(
    "responsive_layouts is a deprecated re-export shim — import from orchestrator.design.responsive_layouts directly",
    DeprecationWarning,
    stacklevel=2,
)
