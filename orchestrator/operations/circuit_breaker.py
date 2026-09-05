"""
Circuit Breaker — Backward-compatibility shim
================================================
The canonical implementation is in orchestrator/circuit_breaker.py — the
copy that used to live here had silently diverged from it, missing the
probe_in_flight fix that prevents concurrent HALF_OPEN probes (BUG-001/
BUG-002) and the already-OPEN no-op guard in record_failure(). Because
orchestrator/operations/resilience.py imports CircuitBreakerOpen from this
module, that divergence also meant its `except CircuitBreakerOpen` clause
was checking against a different, unrelated exception class than the one
every live circuit-breaker construction site (engine_core/container.py,
infrastructure/llm_client.py, engine_slimming.py) actually raises — see
docs/hunts/t5-resilience/inventory.md C1.
"""

from ..circuit_breaker import *  # noqa: F401, F403
