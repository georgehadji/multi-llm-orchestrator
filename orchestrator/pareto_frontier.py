"""Re-export shim — canonical source: orchestrator.analysis.pareto_frontier"""

from orchestrator.analysis.pareto_frontier import *  # noqa: F401, F403

import warnings

warnings.warn(
    "pareto_frontier is a deprecated re-export shim — import from orchestrator.analysis.pareto_frontier directly",
    DeprecationWarning,
    stacklevel=2,
)
