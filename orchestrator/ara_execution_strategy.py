"""Re-export shim — canonical source: orchestrator.reasoning.ara_execution_strategy"""

from orchestrator.reasoning.ara_execution_strategy import *  # noqa: F401, F403

import warnings

warnings.warn(
    "ara_execution_strateg is a deprecated re-export shim — import from orchestrator.reasoning.ara_execution_strategy directly",
    DeprecationWarning,
    stacklevel=2,
)
