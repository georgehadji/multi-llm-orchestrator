"""Re-export shim — canonical source: orchestrator.knowledge.xai_search"""

from orchestrator.knowledge.xai_search import *  # noqa: F401, F403

import warnings

warnings.warn(
    "xai_search is a deprecated re-export shim — import from orchestrator.knowledge.xai_search directly",
    DeprecationWarning,
    stacklevel=2,
)
