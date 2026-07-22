"""Re-export shim — canonical source: orchestrator.integrations.openrouter_ab_testing"""

from orchestrator.integrations.openrouter_ab_testing import *  # noqa: F401, F403

import warnings

warnings.warn(
    "openrouter_ab_testing is a deprecated re-export shim — import from orchestrator.integrations.openrouter_ab_testing directly",
    DeprecationWarning,
    stacklevel=2,
)
