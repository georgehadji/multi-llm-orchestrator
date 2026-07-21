"""Re-export shim — canonical source: orchestrator.meta.config"""

from orchestrator.meta.config import *  # noqa: F401, F403

import warnings

warnings.warn(
    "meta_config is a deprecated re-export shim — import from orchestrator.meta.config directly",
    DeprecationWarning,
    stacklevel=2,
)
