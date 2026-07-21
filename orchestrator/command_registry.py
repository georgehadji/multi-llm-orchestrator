"""Re-export shim — canonical source: orchestrator.commands.registry"""

from orchestrator.commands.registry import *  # noqa: F401, F403

import warnings

warnings.warn(
    "command_registr is a deprecated re-export shim — import from orchestrator.commands.registry directly",
    DeprecationWarning,
    stacklevel=2,
)
