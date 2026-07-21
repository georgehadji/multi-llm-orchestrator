"""Re-export shim — canonical source: orchestrator.quality.code_post_processor"""

from orchestrator.quality.code_post_processor import *  # noqa: F401, F403

import warnings

warnings.warn(
    "code_post_processor is a deprecated re-export shim — import from orchestrator.quality.code_post_processor directly",
    DeprecationWarning,
    stacklevel=2,
)
