"""Re-export shim — canonical source: orchestrator.integrations.multi_tenant_gateway"""

from orchestrator.integrations.multi_tenant_gateway import *  # noqa: F401, F403

import warnings

warnings.warn(
    "multi_tenant_gatewa is a deprecated re-export shim — import from orchestrator.integrations.multi_tenant_gateway directly",
    DeprecationWarning,
    stacklevel=2,
)
