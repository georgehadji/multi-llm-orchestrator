"""Re-export shim — canonical source: orchestrator.plugin.plugin_isolation"""

from orchestrator.plugin.plugin_isolation import *  # noqa: F401, F403

import warnings

warnings.warn(
    "plugin_isolation is a deprecated re-export shim — import from orchestrator.plugin.plugin_isolation directly",
    DeprecationWarning,
    stacklevel=2,
)
