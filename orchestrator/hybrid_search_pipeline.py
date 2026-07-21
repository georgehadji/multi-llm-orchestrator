"""Re-export shim — canonical source: orchestrator.infrastructure.hybrid_search_pipeline"""

from orchestrator.infrastructure.hybrid_search_pipeline import *  # noqa: F401, F403

import warnings

warnings.warn(
    "hybrid_search_pipeline is a deprecated re-export shim — import from orchestrator.infrastructure.hybrid_search_pipeline directly",
    DeprecationWarning,
    stacklevel=2,
)
