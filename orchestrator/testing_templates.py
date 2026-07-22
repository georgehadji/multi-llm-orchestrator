"""Re-export shim — canonical source: orchestrator.generators.testing_templates"""

from orchestrator.generators.testing_templates import *  # noqa: F401, F403

import warnings

warnings.warn(
    "testing_templates is a deprecated re-export shim — import from orchestrator.generators.testing_templates directly",
    DeprecationWarning,
    stacklevel=2,
)
