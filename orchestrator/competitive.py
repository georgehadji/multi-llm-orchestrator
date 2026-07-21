"""Re-export shim — canonical source: orchestrator.analysis.competitive"""

from orchestrator.analysis.competitive import *  # noqa: F401, F403

import warnings

warnings.warn(
    "competitive is a deprecated re-export shim — import from orchestrator.analysis.competitive directly",
    DeprecationWarning,
    stacklevel=2,
)
