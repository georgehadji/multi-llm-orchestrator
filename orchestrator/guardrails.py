"""Re-export shim — canonical source: orchestrator.safety.guardrails"""

from orchestrator.safety.guardrails import *  # noqa: F401, F403

import warnings

warnings.warn(
    "guardrails is a deprecated re-export shim — import from orchestrator.safety.guardrails directly",
    DeprecationWarning,
    stacklevel=2,
)
