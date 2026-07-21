"""Re-export shim — canonical source: orchestrator.quality.adaptive_templates"""

from orchestrator.quality.adaptive_templates import *  # noqa: F401, F403

import warnings

warnings.warn(
    "adaptive_templates is a deprecated re-export shim — import from orchestrator.quality.adaptive_templates directly",
    DeprecationWarning,
    stacklevel=2,
)
