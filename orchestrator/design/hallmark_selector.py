"""
Hallmark Selector — Rule-based design selection engine.
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Given a brief, selects macrostructure, theme, nav, footer, and component
archetypes using deterministic rule-based dispatch (no LLM calls).

Respects the diversification rule: no two consecutive outputs share the same
macrostructure, nav, or footer archetype.
"""

from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING

from orchestrator.models import Genre

from .catalogs import (
    ALL_MACROSTRUCTURE_SLUGS,
    ARCHETYPES,
    DOMAIN_TO_TRIO,
    FOOTER_ROUTING,
    MACROSTRUCTURES,
    NAV_ROUTING,
    THEMES,
    DEFAULT_TRIO,
)

if TYPE_CHECKING:
    from .catalogs import Archetype, Macrostructure, Theme
    from .design_log import DesignLog

logger = logging.getLogger(__name__)


class HallmarkSelector:
    """Rule-based design selection engine.

    Usage:
        selector = HallmarkSelector()
        macro = selector.select_macrostructure(brief, domain, design_log)
        theme = selector.select_theme(brief, macro, preflight)
        nav = selector.select_nav(theme.genre, design_log)
        footer = selector.select_footer(theme.genre, design_log)
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # Macrostructure selection
    # ═══════════════════════════════════════════════════════════════════════════

    def select_macrostructure(
        self,
        brief: str,
        domain: str,
        log: "DesignLog | None",
    ) -> Macrostructure:
        """Select a macrostructure from the domain trio, excluding recently used."""
        # 1. Map domain → trio
        trio_slugs = DOMAIN_TO_TRIO.get(domain.lower(), DEFAULT_TRIO)

        # 2. Exclude recently used
        if log is not None:
            candidates = [s for s in trio_slugs if not log.is_recently_used(s)]
            if not candidates:
                # All trio used — expand to all macrostructures
                candidates = [
                    s for s in ALL_MACROSTRUCTURE_SLUGS if not log.is_recently_used(s)
                ]
            if not candidates:
                # Everything used — just pick from the trio
                candidates = trio_slugs
        else:
            candidates = trio_slugs

        # 3. Pick deterministically (first candidate)
        chosen_slug = candidates[0]
        logger.info(
            "hallmark_selector: macrostructure=%s (from trio %s, excluded=%s)",
            chosen_slug,
            trio_slugs,
            [s for s in trio_slugs if s not in candidates],
        )
        return MACROSTRUCTURES[chosen_slug]

    # ═══════════════════════════════════════════════════════════════════════════
    # Theme selection
    # ═══════════════════════════════════════════════════════════════════════════

    def select_theme(
        self,
        brief: str,
        macrostructure: Macrostructure,
        log: "DesignLog | None",
    ) -> Theme:
        """Select a theme that loves the chosen macrostructure."""
        # 1. Find themes that love this macrostructure
        compatible = [
            t for t in THEMES.values()
            if macrostructure.slug in t.macrostructure_loves
        ]

        # 2. Exclude recently used themes
        if log is not None and compatible:
            candidates = [t for t in compatible if not log.is_theme_recently_used(t.slug)]
            if not candidates:
                candidates = compatible
        else:
            candidates = compatible

        # 3. If no compatible theme, pick any (should not happen with good data)
        if not candidates:
            candidates = list(THEMES.values())
            logger.warning(
                "hallmark_selector: no theme loves %s; falling back to random",
                macrostructure.slug,
            )

        # 4. Pick first compatible (deterministic)
        chosen = candidates[0]
        logger.info(
            "hallmark_selector: theme=%s (genre=%s, compatible=%d)",
            chosen.slug,
            chosen.genre.value,
            len(compatible),
        )
        return chosen

    # ═══════════════════════════════════════════════════════════════════════════
    # Nav / Footer selection
    # ═══════════════════════════════════════════════════════════════════════════

    def select_nav(
        self,
        genre: Genre,
        log: "DesignLog | None",
    ) -> Archetype:
        """Select a nav archetype from the genre routing table."""
        codes = NAV_ROUTING.get(genre.value, ["N1b"])
        return self._select_archetype("nav", codes, log)

    def select_footer(
        self,
        genre: Genre,
        log: "DesignLog | None",
    ) -> Archetype:
        """Select a footer archetype from the genre routing table."""
        codes = FOOTER_ROUTING.get(genre.value, ["Ft1"])
        return self._select_archetype("footer", codes, log)

    def _select_archetype(
        self,
        category: str,
        codes: list[str],
        log: "DesignLog | None",
    ) -> Archetype:
        """Pick first archetype from *codes* not recently used."""
        if log is not None:
            check_fn = (
                log.is_nav_recently_used
                if category == "nav"
                else log.is_footer_recently_used
            )
            candidates = [c for c in codes if not check_fn(c)]
            if not candidates:
                candidates = codes
        else:
            candidates = codes

        chosen_code = candidates[0]
        archetype = ARCHETYPES[chosen_code]
        logger.info(
            "hallmark_selector: %s=%s (%s)",
            category,
            chosen_code,
            archetype.name,
        )
        return archetype

    # ═══════════════════════════════════════════════════════════════════════════
    # Archetype selection for sections
    # ═══════════════════════════════════════════════════════════════════════════

    def select_section_archetypes(
        self,
        macrostructure: Macrostructure,
        section_roles: list[str],
    ) -> list[tuple[Archetype, dict[str, str]]]:
        """Select archetypes for each section role in a page.

        Ensures no two sections share the same archetype on the same page.
        Returns list of (archetype, knob_values).
        """
        used_codes: set[str] = set()
        result: list[tuple[Archetype, dict[str, str]]] = []

        # Map section roles to archetype categories
        category_map: dict[str, str] = {
            "hero": "hero",
            "section_head": "section_head",
            "features": "feature",
            "cta": "cta",
            "testimonials": "testimonial",
            "footer": "footer",
            "nav": "nav",
        }

        for role in section_roles:
            category = category_map.get(role, "feature")
            # Get all archetypes in this category
            available = [
                a for a in ARCHETYPES.values()
                if a.category == category and a.code not in used_codes
            ]
            if not available:
                # All used — reset for this category
                available = [a for a in ARCHETYPES.values() if a.category == category]

            if available:
                # Pick first available (deterministic)
                chosen = available[0]
                used_codes.add(chosen.code)
                # Pick first knob value for each knob (deterministic)
                knob_values = {
                    k: v[0] for k, v in chosen.knobs.items()
                }
                result.append((chosen, knob_values))
            else:
                logger.warning(
                    "hallmark_selector: no archetypes for category %s", category
                )

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Domain inference from brief
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def infer_domain(brief: str) -> str:
        """Infer domain from brief text by keyword matching."""
        lower = brief.lower()
        for domain, trio in DOMAIN_TO_TRIO.items():
            if domain in lower:
                return domain
        # Check for compound matches
        if any(w in lower for w in ["api", "sdk", "cli", "dev", "developer"]):
            return "developer"
        if any(w in lower for w in ["saas", "landing", "marketing"]):
            return "saas"
        if any(w in lower for w in ["portfolio", "agency", "studio"]):
            return "agency"
        if any(w in lower for w in ["shop", "product", "store", "ecommerce"]):
            return "commerce"
        if any(w in lower for w in ["blog", "article", "editorial", "magazine"]):
            return "editorial"
        return "saas"  # fallback
