"""
Crosscutting — Shared Cross-Layer Modules
===========================================
Contains configuration, logging, events, and hooks used across all layers.
"""

from .config import FeatureFlags, OrchestratorSettings, flags, settings

__all__ = ["FeatureFlags", "OrchestratorSettings", "flags", "settings"]
