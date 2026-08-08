"""Readiness probes (Phase 7, P-2/P-3). Aggregated registry for wiring in P-12."""

from __future__ import annotations

from .license_probes import LICENSE_PROBES
from .live_probes import LIVE_PROBES
from .static_probes import STATIC_PROBES
from .supply_chain_probes import SUPPLY_CHAIN_PROBES

ALL_STATIC_PROBES = STATIC_PROBES + SUPPLY_CHAIN_PROBES + LICENSE_PROBES
ALL_PROBES = ALL_STATIC_PROBES + LIVE_PROBES

__all__ = [
    "ALL_PROBES",
    "ALL_STATIC_PROBES",
    "STATIC_PROBES",
    "SUPPLY_CHAIN_PROBES",
    "LICENSE_PROBES",
    "LIVE_PROBES",
]
