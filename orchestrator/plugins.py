"""Re-export shim — canonical source: orchestrator.plugin.plugins"""

from orchestrator.plugin.plugins import *  # noqa: F401, F403

import warnings

warnings.warn(
    "plugins is a deprecated re-export shim — import from orchestrator.plugin.plugins directly",
    DeprecationWarning,
    stacklevel=2,
)
