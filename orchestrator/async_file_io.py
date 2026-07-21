"""Re-export shim — canonical source: orchestrator.events.async_file_io"""

from orchestrator.events.async_file_io import *  # noqa: F401, F403

import warnings

warnings.warn(
    "async_file_io is a deprecated re-export shim — import from orchestrator.events.async_file_io directly",
    DeprecationWarning,
    stacklevel=2,
)
