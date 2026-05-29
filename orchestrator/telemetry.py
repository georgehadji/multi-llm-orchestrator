"""
TelemetryCollector — Backward-compatibility shim
=================================================
Canonical location: orchestrator/infrastructure/telemetry.py
Import from there directly for new code; this shim exists for existing callers.
"""
from .infrastructure.telemetry import TelemetryCollector  # noqa: F401

__all__ = ["TelemetryCollector"]
