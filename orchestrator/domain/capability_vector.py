"""
CapabilityVector — multi-dimensional model capability descriptor.

Phase 0 (ACR WI-0): type definition only. Hydration from OpenRouter
/models payloads and derived-dimension scoring (WI-1+) are out of scope
here — this module exists so ModelProfile.capability can carry an
optional, currently-inert typed field.

Pure domain type: stdlib only, no orchestrator-internal imports, per the
domain-purity import-linter contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityVector:
    """Static/objective dims are sourced from provider metadata (WI-1).
    Derived dims are filled by telemetry/benchmark scoring (WI-1+) and
    default to None until then.
    """

    # ── Static / objective ──
    max_context: int
    vision: bool
    tool_use: bool
    json_mode: bool
    price: float

    # ── Derived — never hand-typed ──
    reasoning: float | None = None
    legal: float | None = None
    coding: float | None = None
    citation: float | None = None
