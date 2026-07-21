"""Re-export shim — canonical source: orchestrator.quality.browser_testing"""

from orchestrator.quality.browser_testing import *  # noqa: F401, F403

import warnings

warnings.warn(
    "browser_testing is a deprecated re-export shim — import from orchestrator.quality.browser_testing directly",
    DeprecationWarning,
    stacklevel=2,
)
