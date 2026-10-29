"""Re-export shim - canonical source: orchestrator.analysis.progress_writer"""

from orchestrator.analysis.progress_writer import *  # noqa: F401, F403

import warnings

warnings.warn(
    "progress_writer is a deprecated re-export shim - import from orchestrator.analysis.progress_writer directly",
    DeprecationWarning,
    stacklevel=2,
)
