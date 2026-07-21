"""Re-export shim — canonical source: orchestrator.security.wordpress_plugin_rules"""

from orchestrator.security.wordpress_plugin_rules import *  # noqa: F401, F403

import warnings

warnings.warn(
    "wordpress_plugin_rules is a deprecated re-export shim — import from orchestrator.security.wordpress_plugin_rules directly",
    DeprecationWarning,
    stacklevel=2,
)
