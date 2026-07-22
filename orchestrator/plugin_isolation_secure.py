"""Re-export shim — canonical source: orchestrator.plugin.plugin_isolation_secure"""

from orchestrator.plugin.plugin_isolation_secure import *  # noqa: F401, F403

import warnings

warnings.warn(
    "plugin_isolation_secure is a deprecated re-export shim — import from orchestrator.plugin.plugin_isolation_secure directly",
    DeprecationWarning,
    stacklevel=2,
)
