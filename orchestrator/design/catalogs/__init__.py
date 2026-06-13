"""
Hallmark Design Catalogs — Reference data for structural design generation.
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

This package contains immutable reference data:
- 21 macrostructures (page shapes)
- 20 catalog themes (+ custom branch)
- 50 component archetypes with variation knobs
- Domain→trio routing tables
- Nav/footer archetype routing tables

Usage:
    from orchestrator.design.catalogs import MACROSTRUCTURES, THEMES, ARCHETYPES
    from orchestrator.design.catalogs import DOMAIN_TO_TRIO, NAV_ROUTING, FOOTER_ROUTING
"""

from __future__ import annotations

from .macrostructures import MACROSTRUCTURES, Macrostructure, ALL_MACROSTRUCTURE_SLUGS
from .themes import THEMES, Theme, Genre
from .archetypes import ARCHETYPES, Archetype
from .routing import DOMAIN_TO_TRIO, NAV_ROUTING, FOOTER_ROUTING, DEFAULT_TRIO

__all__ = [
    "MACROSTRUCTURES",
    "Macrostructure",
    "ALL_MACROSTRUCTURE_SLUGS",
    "THEMES",
    "Theme",
    "Genre",
    "ARCHETYPES",
    "Archetype",
    "DOMAIN_TO_TRIO",
    "DEFAULT_TRIO",
    "NAV_ROUTING",
    "FOOTER_ROUTING",
]
