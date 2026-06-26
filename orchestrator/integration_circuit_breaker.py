"""
integration_circuit_breaker — Backward-compatibility shim
The canonical implementation lives in orchestrator/integrations/integration_circuit_breaker.py.
New code should import from `orchestrator.integrations.integration_circuit_breaker` directly.
"""

from .integrations.integration_circuit_breaker import *  # noqa: F401, F403
