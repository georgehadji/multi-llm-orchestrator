"""Static readiness probes (Phase 7, P-2). Aggregated registry for wiring in P-12."""

from __future__ import annotations

from .license_probes import LICENSE_PROBES
from .static_probes import STATIC_PROBES
from .supply_chain_probes import SUPPLY_CHAIN_PROBES

ALL_STATIC_PROBES = STATIC_PROBES + SUPPLY_CHAIN_PROBES + LICENSE_PROBES

__all__ = ["ALL_STATIC_PROBES", "STATIC_PROBES", "SUPPLY_CHAIN_PROBES", "LICENSE_PROBES"]
