"""Re-export shim — canonical source: orchestrator.safety.security_templates"""

from orchestrator.safety.security_templates import *  # noqa: F401, F403

import warnings

warnings.warn(
    "security_templates is a deprecated re-export shim — import from orchestrator.safety.security_templates directly",
    DeprecationWarning,
    stacklevel=2,
)
