"""Re-export shim — canonical source: orchestrator.reasoning.brain"""

from orchestrator.reasoning.brain import *  # noqa: F401, F403

import warnings

warnings.warn(
    "brain is a deprecated re-export shim — import from orchestrator.reasoning.brain directly",
    DeprecationWarning,
    stacklevel=2,
)
