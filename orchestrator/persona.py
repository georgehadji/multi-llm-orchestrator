"""Re-export shim — canonical source: orchestrator.agents.persona"""

from orchestrator.agents.persona import *  # noqa: F401, F403

import warnings

warnings.warn(
    "persona is a deprecated re-export shim — import from orchestrator.agents.persona directly",
    DeprecationWarning,
    stacklevel=2,
)
