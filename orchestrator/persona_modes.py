"""Re-export shim — canonical source: orchestrator.agents.persona_modes"""

from orchestrator.agents.persona_modes import *  # noqa: F401, F403

import warnings

warnings.warn(
    "persona_modes is a deprecated re-export shim — import from orchestrator.agents.persona_modes directly",
    DeprecationWarning,
    stacklevel=2,
)
