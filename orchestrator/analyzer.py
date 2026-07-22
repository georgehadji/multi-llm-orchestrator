"""Re-export shim — canonical source: orchestrator.analysis.analyzer"""

from orchestrator.analysis.analyzer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "analyzer is a deprecated re-export shim — import from orchestrator.analysis.analyzer directly",
    DeprecationWarning,
    stacklevel=2,
)
