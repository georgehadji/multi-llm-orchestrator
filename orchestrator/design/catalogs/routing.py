"""
Routing Tables — Domain → Macrostructure Trio + Genre → Nav/Footer.
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Determines which macrostructures, nav archetypes, and footer archetypes
are appropriate for a given domain or genre.

Source: Hallmark design skill (references/structure.md, component-cookbook.md)
"""

from __future__ import annotations

from orchestrator.models import Genre

# ═══════════════════════════════════════════════════════════════════════════════
# Domain → Trio (offer these three; never default)
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_TO_TRIO: dict[str, list[str]] = {
    # Media / audio
    "podcast": ["photographic", "quote_led", "letter"],
    "audio": ["photographic", "quote_led", "letter"],
    "music": ["photographic", "quote_led", "letter"],
    "playlist": ["photographic", "quote_led", "letter"],
    "listening": ["photographic", "quote_led", "letter"],
    # Commerce
    "shop": ["catalogue", "photographic", "bento_grid"],
    "store": ["catalogue", "photographic", "bento_grid"],
    "product": ["catalogue", "photographic", "bento_grid"],
    "merch": ["catalogue", "photographic", "bento_grid"],
    "commerce": ["catalogue", "photographic", "bento_grid"],
    "ecom": ["catalogue", "photographic", "bento_grid"],
    # Developer / docs
    "docs": ["workbench", "long_document", "component_playground"],
    "cli": ["workbench", "long_document", "component_playground"],
    "sdk": ["workbench", "long_document", "component_playground"],
    "api": ["workbench", "stat_led", "component_playground"],
    "library": ["workbench", "long_document", "component_playground"],
    "open-source": ["workbench", "index_first", "component_playground"],
    "developer": ["workbench", "stat_led", "component_playground"],
    # Platform / SaaS / B2B
    "platform": ["bento_grid", "workbench", "stat_led"],
    "infra": ["bento_grid", "workbench", "stat_led"],
    "observability": ["bento_grid", "workbench", "stat_led"],
    "dashboard": ["bento_grid", "workbench", "stat_led"],
    "saas": ["bento_grid", "workbench", "stat_led"],
    "b2b": ["bento_grid", "workbench", "stat_led"],
    "b2b-tool": ["bento_grid", "workbench", "stat_led"],
    # Agency / creative
    "agency": ["portfolio_grid", "split_studio", "index_first"],
    "studio": ["portfolio_grid", "split_studio", "index_first"],
    "work-led": ["portfolio_grid", "split_studio", "index_first"],
    "case-studies": ["portfolio_grid", "split_studio", "index_first"],
    "portfolio": ["portfolio_grid", "split_studio", "index_first"],
    "freelance": ["portfolio_grid", "split_studio", "index_first"],
    "creative": ["portfolio_grid", "split_studio", "index_first"],
    # Personal
    "personal": ["long_document", "letter", "index_first"],
    "one-pager": ["long_document", "letter", "index_first"],
    "about-me": ["long_document", "letter", "index_first"],
    "resume": ["long_document", "letter", "index_first"],
    "individual": ["long_document", "letter", "index_first"],
    # Food / hospitality
    "restaurant": ["photographic", "long_document", "catalogue"],
    "cafe": ["photographic", "long_document", "catalogue"],
    "bar": ["photographic", "long_document", "catalogue"],
    "food": ["photographic", "long_document", "catalogue"],
    "kitchen": ["photographic", "long_document", "catalogue"],
    "menu": ["photographic", "long_document", "catalogue"],
    # Fashion / beauty
    "fashion": ["photographic", "catalogue", "marquee_hero"],
    "apparel": ["photographic", "catalogue", "marquee_hero"],
    "beauty": ["photographic", "catalogue", "marquee_hero"],
    "lookbook": ["photographic", "catalogue", "marquee_hero"],
    # Fintech
    "fintech": ["stat_led", "workbench", "long_document"],
    "banking": ["stat_led", "workbench", "long_document"],
    "payments": ["stat_led", "workbench", "long_document"],
    "invest": ["stat_led", "workbench", "long_document"],
    "trading": ["stat_led", "workbench", "long_document"],
    # Campaign / cause
    "manifesto": ["manifesto", "quote_led", "stat_led"],
    "campaign": ["manifesto", "quote_led", "stat_led"],
    "cause": ["manifesto", "quote_led", "stat_led"],
    "advocacy": ["manifesto", "quote_led", "stat_led"],
    "political": ["manifesto", "quote_led", "stat_led"],
    # Editorial / publishing
    "editorial": ["specimen", "long_document", "type_specimen"],
    "foundry": ["specimen", "long_document", "type_specimen"],
    "magazine": ["specimen", "long_document", "type_specimen"],
    "type": ["specimen", "long_document", "type_specimen"],
    "specimen": ["specimen", "long_document", "type_specimen"],
    # Product launch / marketing
    "product-launch": ["bento_grid", "workbench", "stat_led"],
    "marketing": ["bento_grid", "workbench", "stat_led"],
    # Event / conference
    "conference": ["marquee_hero", "manifesto", "photographic"],
    "event": ["marquee_hero", "manifesto", "photographic"],
    "speaker": ["marquee_hero", "manifesto", "photographic"],
    "keynote": ["marquee_hero", "manifesto", "photographic"],
}

# Fallback when no domain word matches
DEFAULT_TRIO: list[str] = ["bento_grid", "long_document", "manifesto"]


# ═══════════════════════════════════════════════════════════════════════════════
# Nav routing — which nav fits which genre
# ═══════════════════════════════════════════════════════════════════════════════

NAV_ROUTING: dict[str, list[str]] = {
    Genre.EDITORIAL.value: ["N6", "N1a", "N9", "N12"],
    Genre.MODERN_MINIMAL.value: ["N1b", "N5", "N11", "N13", "N9"],
    Genre.ATMOSPHERIC.value: ["N5", "N9", "N4", "N13", "N1b"],
    Genre.PLAYFUL.value: ["N1b", "N5", "N11", "N12", "N13", "N7"],
    Genre.TERMINAL.value: ["N8", "N4", "N13"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Footer routing — which footer fits which genre
# ═══════════════════════════════════════════════════════════════════════════════

FOOTER_ROUTING: dict[str, list[str]] = {
    Genre.EDITORIAL.value: ["Ft1", "Ft2", "Ft4", "Ft6", "Ft7"],
    Genre.MODERN_MINIMAL.value: ["Ft2", "Ft1", "Ft5"],
    Genre.ATMOSPHERIC.value: ["Ft5", "Ft1", "Ft2"],
    Genre.PLAYFUL.value: ["Ft8", "Ft5", "Ft3"],
    Genre.TERMINAL.value: ["Ft4", "Ft2"],
}
