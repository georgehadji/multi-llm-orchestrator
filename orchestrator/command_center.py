"""Re-export shim — canonical source: orchestrator.commands.center"""

from orchestrator.commands.center import *  # noqa: F401, F403

import warnings

warnings.warn(
    "command_center is a deprecated re-export shim — import from orchestrator.commands.center directly",
    DeprecationWarning,
    stacklevel=2,
)
