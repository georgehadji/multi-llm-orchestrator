"""Re-export shim — canonical source: orchestrator.quality.benchmark_suite"""

from orchestrator.quality.benchmark_suite import *  # noqa: F401, F403

import warnings

warnings.warn(
    "benchmark_suite is a deprecated re-export shim — import from orchestrator.quality.benchmark_suite directly",
    DeprecationWarning,
    stacklevel=2,
)
