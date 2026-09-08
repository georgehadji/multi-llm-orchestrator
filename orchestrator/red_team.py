"""Re-export shim — canonical source: orchestrator.safety.red_team"""

from orchestrator.safety.red_team import *  # noqa: F401, F403

import warnings

warnings.warn(
    "red_team is a deprecated re-export shim — import from orchestrator.safety.red_team directly",
    DeprecationWarning,
    stacklevel=2,
)
