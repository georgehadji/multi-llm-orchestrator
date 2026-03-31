"""
Component Registry for DSDG (Design-System-Driven Generation)
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Multi-source component library with quality scoring and automated selection
based on design system compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .design_system import (
    ComponentSource,
    ComponentSpec,
    DesignSystem,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Component Library Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-defined component templates from various sources
# In production, these would be fetched from URLs or npm packages
COMPONENT_LIBRARY: dict[str, list[ComponentSpec]] = {
    # ═══════════════════════════════════════════════════════════════════════════
    # HERO SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "hero": [
        ComponentSpec(
            name="hero_gradient",
            source=ComponentSource.ACETERNITY,
            category="hero",
            variant="gradient",
            layout_type="flex",
            primary_color=None,  # Will be set from design system
            animation_style="bold_creative",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
A modern hero section with animated gradient background.
Features: Large heading, subheading, CTA buttons, optional image/illustration.
Animation: Subtle gradient movement, fade-in on scroll.
Layout: Centered content, max-width constrained.
Mobile: Stacks vertically, reduced animation.
""",
        ),
        ComponentSpec(
            name="hero_minimal",
            source=ComponentSource.SHADCN,
            category="hero",
            variant="minimal",
            layout_type="flex",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference="""
Clean, minimal hero section with generous whitespace.
Features: Bold typography, single CTA, subtle divider.
Animation: None or very subtle fade-in.
Layout: Centered or left-aligned content.
Mobile: Maintains whitespace ratios.
""",
        ),
        ComponentSpec(
            name="hero_split",
            source=ComponentSource.TWENTYFIRST,
            category="hero",
            variant="split",
            layout_type="grid",
            primary_color=None,
            animation_style="saas_modern",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Split-screen hero with content on left, image/visual on right.
Features: Two-column grid, sticky visual element.
Animation: Slide-in from sides.
Layout: 50/50 split on desktop, stacks on mobile.
Mobile: Content first, visual below.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURES SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "features": [
        ComponentSpec(
            name="features_grid",
            source=ComponentSource.SHADCN,
            category="features",
            variant="grid",
            layout_type="grid",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Feature grid with cards in responsive grid layout.
Features: 3-4 columns, icon + title + description per card.
Animation: Hover lift effect, subtle scale.
Layout: 12-column grid, auto-fit columns.
Mobile: Single column, full-width cards.
""",
        ),
        ComponentSpec(
            name="features_alternating",
            source=ComponentSource.ACETERNITY,
            category="features",
            variant="alternating",
            layout_type="flex",
            primary_color=None,
            animation_style="bold_creative",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference="""
Alternating feature sections with image/text on opposite sides.
Features: Zig-zag layout, large images, detailed descriptions.
Animation: Fade-in on scroll, parallax images.
Layout: Full-width sections, alternating content order.
Mobile: Always image on top, text below.
""",
        ),
        ComponentSpec(
            name="features_bento",
            source=ComponentSource.MAGIC_UI,
            category="features",
            variant="bento",
            layout_type="grid",
            primary_color=None,
            animation_style="saas_modern",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Bento box style feature grid with varying card sizes.
Features: Masonry-like layout, some cards span multiple rows/columns.
Animation: Staggered fade-in, hover glow effects.
Layout: CSS grid with named areas.
Mobile: Linearized, all cards same width.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # PRICING SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "pricing": [
        ComponentSpec(
            name="pricing_cards",
            source=ComponentSource.SHADCN,
            category="pricing",
            variant="cards",
            layout_type="flex",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Pricing cards with tier comparison.
Features: 3 tiers (Basic, Pro, Enterprise), feature lists, CTA buttons.
Animation: Hover scale on cards, recommended badge pulse.
Layout: 3 cards in row, middle card highlighted.
Mobile: Stacked cards, swipeable.
""",
        ),
        ComponentSpec(
            name="pricing_toggle",
            source=ComponentSource.TWENTYFIRST,
            category="pricing",
            variant="toggle",
            layout_type="flex",
            primary_color=None,
            animation_style="saas_modern",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Pricing with monthly/yearly toggle switch.
Features: Toggle switch with animation, discounted yearly pricing.
Animation: Smooth toggle transition, price count-up animation.
Layout: Toggle centered above pricing cards.
Mobile: Toggle above stacked cards.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # TESTIMONIALS SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "testimonials": [
        ComponentSpec(
            name="testimonials_grid",
            source=ComponentSource.SHADCN,
            category="testimonials",
            variant="grid",
            layout_type="grid",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference="""
Testimonial grid with customer quotes.
Features: Avatar, name, role, company, quote text, stars.
Animation: Fade-in on scroll.
Layout: 3-column grid.
Mobile: Single column carousel.
""",
        ),
        ComponentSpec(
            name="testimonials_slider",
            source=ComponentSource.ACETERNITY,
            category="testimonials",
            variant="slider",
            layout_type="flex",
            primary_color=None,
            animation_style="bold_creative",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Testimonial slider/carousel with auto-play.
Features: One testimonial at a time, navigation dots, auto-rotate.
Animation: Smooth slide transitions, fade effect.
Layout: Centered content, full-width background.
Mobile: Touch-enabled swipe.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # FAQ SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "faq": [
        ComponentSpec(
            name="faq_accordion",
            source=ComponentSource.SHADCN,
            category="faq",
            variant="accordion",
            layout_type="stack",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
FAQ accordion with expandable items.
Features: Click to expand/collapse, smooth animation, keyboard navigation.
Animation: Chevron rotate, content slide down.
Layout: Centered column, max-width constrained.
Mobile: Full-width, touch-friendly.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # CTA SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "cta": [
        ComponentSpec(
            name="cta_banner",
            source=ComponentSource.SHADCN,
            category="cta",
            variant="banner",
            layout_type="flex",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference="""
Call-to-action banner section.
Features: Bold heading, subheading, primary and secondary CTAs.
Animation: Subtle hover on buttons.
Layout: Centered content, full-width background.
Mobile: Stacked buttons.
""",
        ),
        ComponentSpec(
            name="cta_final",
            source=ComponentSource.TWENTYFIRST,
            category="cta",
            variant="final",
            layout_type="flex",
            primary_color=None,
            animation_style="bold_creative",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Final CTA section before footer.
Features: Large heading, compelling copy, single focused CTA.
Animation: Gradient background animation.
Layout: Centered, full-bleed background.
Mobile: Reduced padding, single button.
""",
        ),
    ],
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "footer": [
        ComponentSpec(
            name="footer_standard",
            source=ComponentSource.SHADCN,
            category="footer",
            variant="standard",
            layout_type="grid",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=True,
            prompt_reference="""
Standard footer with multiple columns.
Features: Logo, navigation links, social icons, copyright, legal links.
Animation: Hover on links.
Layout: 4-column grid on desktop.
Mobile: Stacked columns, hamburger menu for links.
""",
        ),
        ComponentSpec(
            name="footer_minimal",
            source=ComponentSource.SHADCN,
            category="footer",
            variant="minimal",
            layout_type="flex",
            primary_color=None,
            animation_style="premium_minimal",
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference="""
Minimal footer with simple layout.
Features: Copyright, basic links, social icons.
Animation: None.
Layout: Centered or split (links left, social right).
Mobile: Single column.
""",
        ),
    ],
}


@dataclass
class ComponentRegistry:
    """
    Multi-source component library with quality scoring.

    Selects best-matching components based on:
    - Design system compatibility
    - Page type requirements
    - Section requirements
    """

    SOURCES: dict[str, str] = None

    def __post_init__(self):
        if self.SOURCES is None:
            self.SOURCES = {
                "twentyfirst": "https://twentyfirst.dev",
                "shadcn": "https://ui.shadcn.com",
                "aceternity": "https://ui.aceternity.com",
                "magic_ui": "https://magicui.design",
                "curated_internal": "./components/curated/",
            }

    async def select_components(
        self,
        page_type: str,
        design_system: DesignSystem,
        sections_needed: list[str],
    ) -> list[ComponentSpec]:
        """
        Select best-matching components from multiple sources.

        Parameters
        ----------
        page_type : Type of page ("landing", "saas", "portfolio", "ecommerce")
        design_system : Design system to match against
        sections_needed : List of section names needed

        Returns
        -------
        List of selected ComponentSpec objects
        """
        selected = []

        for section in sections_needed:
            candidates = self._get_candidates(section)

            if not candidates:
                # Create a default component spec for unknown sections
                logger.warning(f"No component templates for section: {section}")
                default = self._create_default_component(section, design_system)
                selected.append(default)
                continue

            # Score each candidate against design system compatibility
            scored: list[tuple[ComponentSpec, float]] = []
            for candidate in candidates:
                # Apply design system colors to component
                candidate.primary_color = design_system.colors.primary
                score = candidate.compatibility_score(design_system)
                scored.append((candidate, score))

            # Pick highest-scoring
            best = max(scored, key=lambda x: x[1])
            logger.info(f"Selected {best[0].name} for {section} (score: {best[1]:.2f})")
            selected.append(best[0])

        return selected

    def _get_candidates(self, section: str) -> list[ComponentSpec]:
        """Get candidate components for a section."""
        return COMPONENT_LIBRARY.get(section, [])

    def _create_default_component(
        self,
        section: str,
        design_system: DesignSystem,
    ) -> ComponentSpec:
        """Create a default component spec for unknown sections."""
        return ComponentSpec(
            name=f"{section}_default",
            source=ComponentSource.CUSTOM,
            category=section,
            variant="default",
            layout_type="flex",
            primary_color=design_system.colors.primary,
            animation_style=design_system.tone.value,
            has_aria_labels=True,
            keyboard_navigable=True,
            responsive=True,
            dark_mode_support=False,
            prompt_reference=f"""
Default {section} section following design system.
Features: Standard layout for {section} content.
Animation: Subtle, respects design system.
Layout: Responsive, mobile-first.
Colors: Uses design system palette.
""",
        )

    def get_component_prompt(
        self,
        component: ComponentSpec,
        design_system: DesignSystem,
        content_brief: dict | None = None,
    ) -> str:
        """
        Generate a prompt for creating a component.

        Parameters
        ----------
        component : Component spec to generate
        design_system : Design system to follow
        content_brief : Optional content for the section

        Returns
        -------
        Prompt string for LLM code generation
        """
        content = content_brief or {}

        return f"""
COMPONENT REFERENCE:
Name: {component.name}
Source: {component.source.value}
Category: {component.category}
Variant: {component.variant}

DESCRIPTION:
{component.prompt_reference}

CONTENT FOR THIS SECTION:
{content.get('headline', '')}
{content.get('cta', '')}
{content.get('pain_points', [])}

DESIGN SYSTEM TOKENS TO USE:
{design_system.to_prompt_context()}

Generate a React/Next.js component with Tailwind CSS.
Use ONLY design system tokens for colors, spacing, typography.
Include: semantic HTML, aria-labels, focus states, reduced-motion support.
Mobile-first responsive design.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════


def get_registry() -> ComponentRegistry:
    """Get or create component registry singleton."""
    if not hasattr(get_registry, "_registry"):
        get_registry._registry = ComponentRegistry()
    return get_registry._registry
