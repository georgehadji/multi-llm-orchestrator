"""Re-export shim — canonical source: orchestrator.learning.learning_aggregator"""

from orchestrator.learning.learning_aggregator import *  # noqa: F401, F403

import warnings

warnings.warn(
    "learning_aggregator is a deprecated re-export shim — import from orchestrator.learning.learning_aggregator directly",
    DeprecationWarning,
    stacklevel=2,
)
