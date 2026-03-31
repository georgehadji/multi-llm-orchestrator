"""
Design System Models for DSDG (Design-System-Driven Generation)
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Core data models for design system definition, component selection,
and quality validation in AI-generated websites and web apps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Brand Tone Presets
# ═══════════════════════════════════════════════════════════════════════════════


class BrandTone(str, Enum):
    """Brand tone presets for quick start."""

    PREMIUM_MINIMAL = "premium_minimal"
    BOLD_CREATIVE = "bold_creative"
    CORPORATE_CLEAN = "corporate_clean"
    PLAYFUL_MODERN = "playful_modern"
    LUXURY_ELEGANT = "luxury_elegant"
    MEDICAL_TRUST = "medical_trust"
    SAAS_MODERN = "saas_modern"
    STARTUP_BOLD = "startup_bold"


# ═══════════════════════════════════════════════════════════════════════════════
# Design System Presets (from DESIGN.md)
# ═══════════════════════════════════════════════════════════════════════════════

DESIGN_PRESETS: dict[BrandTone, dict] = {
    BrandTone.PREMIUM_MINIMAL: {
        "fonts": ("Plus Jakarta Sans", "Inter"),
        "primary": "#0f172a",
        "accent": "#6366f1",
        "style": "Clean lines, generous whitespace, subtle shadows",
        "surface": "#fafafa",
        "text_primary": "#0f172a",
        "text_secondary": "#6b7280",
    },
    BrandTone.BOLD_CREATIVE: {
        "fonts": ("Cabinet Grotesk", "Satoshi"),
        "primary": "#18181b",
        "accent": "#f97316",
        "style": "High contrast, bold typography, dynamic animations",
        "surface": "#ffffff",
        "text_primary": "#18181b",
        "text_secondary": "#71717a",
    },
    BrandTone.CORPORATE_CLEAN: {
        "fonts": ("Inter", "System UI"),
        "primary": "#1e40af",
        "accent": "#3b82f6",
        "style": "Professional, trustworthy, clean layouts",
        "surface": "#ffffff",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
    },
    BrandTone.PLAYFUL_MODERN: {
        "fonts": ("Poppins", "Nunito"),
        "primary": "#7c3aed",
        "accent": "#ec4899",
        "style": "Vibrant colors, rounded corners, friendly animations",
        "surface": "#faf5ff",
        "text_primary": "#581c87",
        "text_secondary": "#a855f7",
    },
    BrandTone.LUXURY_ELEGANT: {
        "fonts": ("Playfair Display", "Lora"),
        "primary": "#1c1917",
        "accent": "#d4a853",
        "style": "Serif elegance, dark backgrounds, refined spacing",
        "surface": "#0c0a09",
        "text_primary": "#f5f5f4",
        "text_secondary": "#a8a29e",
    },
    BrandTone.MEDICAL_TRUST: {
        "fonts": ("DM Sans", "Inter"),
        "primary": "#0c4a6e",
        "accent": "#06b6d4",
        "style": "Trust-building, clean, accessible, calming colors",
        "surface": "#f0f9ff",
        "text_primary": "#0c4a6e",
        "text_secondary": "#64748b",
    },
    BrandTone.SAAS_MODERN: {
        "fonts": ("Geist", "Geist Mono"),
        "primary": "#020617",
        "accent": "#8b5cf6",
        "style": "Dark/light mode, glassmorphism, gradient accents",
        "surface": "#0f172a",
        "text_primary": "#f8fafc",
        "text_secondary": "#94a3b8",
    },
    BrandTone.STARTUP_BOLD: {
        "fonts": ("Clash Display", "Inter"),
        "primary": "#000000",
        "accent": "#22c55e",
        "style": "Bold typography, high contrast, energetic",
        "surface": "#ffffff",
        "text_primary": "#000000",
        "text_secondary": "#525252",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Design System Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TypographyScale:
    """Typography scale definition."""

    xs: str = "0.75rem"  # 12px
    sm: str = "0.875rem"  # 14px
    base: str = "1rem"  # 16px
    lg: str = "1.25rem"  # 20px
    xl: str = "1.563rem"  # 25px
    xl2: str = "1.953rem"  # 31px
    xl3: str = "2.441rem"  # 39px
    xl4: str = "3.052rem"  # 49px


@dataclass
class Colors:
    """Color palette definition."""

    primary: str = "#1a1a2e"
    accent: str = "#e94560"
    surface: str = "#fafafa"
    surface_alt: str = "#f0f0f5"
    text_primary: str = "#1a1a2e"
    text_secondary: str = "#6b7280"
    border: str = "#e5e7eb"
    success: str = "#10b981"
    error: str = "#ef4444"
    warning: str = "#f59e0b"
    info: str = "#3b82f6"


@dataclass
class Spacing:
    """Spacing scale definition."""

    unit: str = "0.25rem"  # 4px base
    scale: list[str] = field(
        default_factory=lambda: [
            "0",
            "0.25rem",
            "0.5rem",
            "0.75rem",
            "1rem",
            "1.5rem",
            "2rem",
            "3rem",
            "4rem",
            "6rem",
            "8rem",
            "12rem",
            "16rem",
        ]
    )


@dataclass
class Layout:
    """Layout configuration."""

    max_width: str = "1280px"
    columns: int = 12
    gutter: str = "2rem"
    breakpoints: dict[str, str] = field(
        default_factory=lambda: {
            "sm": "640px",
            "md": "768px",
            "lg": "1024px",
            "xl": "1280px",
            "2xl": "1536px",
        }
    )


@dataclass
class BorderRadius:
    """Border radius tokens."""

    sm: str = "0.375rem"
    md: str = "0.75rem"
    lg: str = "1rem"
    xl: str = "1.5rem"
    full: str = "9999px"


@dataclass
class Shadow:
    """Shadow tokens."""

    sm: str = "0 1px 2px rgba(0,0,0,0.05)"
    md: str = "0 4px 6px rgba(0,0,0,0.07)"
    lg: str = "0 10px 15px rgba(0,0,0,0.1)"
    xl: str = "0 20px 25px rgba(0,0,0,0.15)"
    inner: str = "inset 0 2px 4px rgba(0,0,0,0.06)"


@dataclass
class Animation:
    """Animation tokens."""

    duration: str = "300ms"
    duration_slow: str = "500ms"
    duration_fast: str = "150ms"
    easing: str = "cubic-bezier(0.4, 0, 0.2, 1)"
    easing_in: str = "cubic-bezier(0.4, 0, 1, 1)"
    easing_out: str = "cubic-bezier(0, 0, 0.2, 1)"
    easing_in_out: str = "cubic-bezier(0.4, 0, 0.2, 1)"


@dataclass
class Accessibility:
    """Accessibility configuration."""

    min_contrast_ratio: float = 4.5
    focus_ring: bool = True
    reduced_motion_support: bool = True
    semantic_html: bool = True
    skip_links: bool = True
    aria_labels: bool = True


@dataclass
class DesignSystem:
    """
    Complete design system definition.

    This is the core data structure that drives all UI generation.
    Every component generated MUST use values from this system.
    """

    # Brand identity
    brand_name: str = ""
    industry: str = ""
    tone: BrandTone = BrandTone.PREMIUM_MINIMAL

    # Typography
    font_heading: str = "Plus Jakarta Sans"
    font_body: str = "Inter"
    typography: TypographyScale = field(default_factory=TypographyScale)

    # Colors
    colors: Colors = field(default_factory=Colors)

    # Spacing & Layout
    spacing: Spacing = field(default_factory=Spacing)
    layout: Layout = field(default_factory=Layout)

    # Component tokens
    border_radius: BorderRadius = field(default_factory=BorderRadius)
    shadow: Shadow = field(default_factory=Shadow)
    animation: Animation = field(default_factory=Animation)

    # Accessibility
    accessibility: Accessibility = field(default_factory=Accessibility)

    # Style description for LLM prompts
    style_description: str = ""

    @classmethod
    def from_preset(cls, tone: BrandTone) -> DesignSystem:
        """Create a design system from a preset."""
        preset = DESIGN_PRESETS.get(tone, DESIGN_PRESETS[BrandTone.PREMIUM_MINIMAL])

        fonts = preset.get("fonts", ("Plus Jakarta Sans", "Inter"))

        return cls(
            brand_name="",
            industry="",
            tone=tone,
            font_heading=fonts[0],
            font_body=fonts[1],
            colors=Colors(
                primary=preset.get("primary", "#1a1a2e"),
                accent=preset.get("accent", "#e94560"),
                surface=preset.get("surface", "#fafafa"),
                text_primary=preset.get("text_primary", "#1a1a2e"),
                text_secondary=preset.get("text_secondary", "#6b7280"),
            ),
            style_description=preset.get("style", ""),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> DesignSystem:
        """Load design system from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> DesignSystem:
        """Create DesignSystem from dictionary."""
        brand = data.get("brand", {})
        typography_data = data.get("typography", {})
        colors_data = data.get("colors", {})
        spacing_data = data.get("spacing", {})
        layout_data = data.get("layout", {})
        components_data = data.get("components", {})
        accessibility_data = data.get("accessibility", {})

        # Parse tone
        tone_str = brand.get("tone", "premium_minimal")
        try:
            tone = BrandTone(tone_str)
        except ValueError:
            tone = BrandTone.PREMIUM_MINIMAL

        return cls(
            brand_name=brand.get("name", ""),
            industry=brand.get("industry", ""),
            tone=tone,
            font_heading=typography_data.get("font_heading", "Plus Jakarta Sans"),
            font_body=typography_data.get("font_body", "Inter"),
            typography=TypographyScale(**typography_data.get("scale", {})),
            colors=Colors(**colors_data),
            spacing=Spacing(**spacing_data),
            layout=Layout(**layout_data),
            border_radius=BorderRadius(**components_data.get("border_radius", {})),
            shadow=Shadow(**components_data.get("shadow", {})),
            animation=Animation(**components_data.get("animation", {})),
            accessibility=Accessibility(**accessibility_data),
        )

    def to_yaml(self) -> str:
        """Export design system to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Export design system to dictionary."""
        return {
            "brand": {
                "name": self.brand_name,
                "industry": self.industry,
                "tone": self.tone.value,
            },
            "typography": {
                "font_heading": self.font_heading,
                "font_body": self.font_body,
                "scale": {
                    "xs": self.typography.xs,
                    "sm": self.typography.sm,
                    "base": self.typography.base,
                    "lg": self.typography.lg,
                    "xl": self.typography.xl,
                    "2xl": self.typography.xl2,
                    "3xl": self.typography.xl3,
                    "4xl": self.typography.xl4,
                },
            },
            "colors": {
                "primary": self.colors.primary,
                "accent": self.colors.accent,
                "surface": self.colors.surface,
                "surface_alt": self.colors.surface_alt,
                "text_primary": self.colors.text_primary,
                "text_secondary": self.colors.text_secondary,
                "border": self.colors.border,
                "success": self.colors.success,
                "error": self.colors.error,
            },
            "spacing": {
                "unit": self.spacing.unit,
                "scale": self.spacing.scale,
            },
            "layout": {
                "max_width": self.layout.max_width,
                "columns": self.layout.columns,
                "gutter": self.layout.gutter,
                "breakpoints": self.layout.breakpoints,
            },
            "components": {
                "border_radius": {
                    "sm": self.border_radius.sm,
                    "md": self.border_radius.md,
                    "lg": self.border_radius.lg,
                    "full": self.border_radius.full,
                },
                "shadow": {
                    "sm": self.shadow.sm,
                    "md": self.shadow.md,
                    "lg": self.shadow.lg,
                },
                "animation": {
                    "duration": self.animation.duration,
                    "easing": self.animation.easing,
                },
            },
            "accessibility": {
                "min_contrast_ratio": self.accessibility.min_contrast_ratio,
                "focus_ring": self.accessibility.focus_ring,
                "reduced_motion_support": self.accessibility.reduced_motion_support,
                "semantic_html": self.accessibility.semantic_html,
            },
        }

    def to_prompt_context(self) -> str:
        """Convert design system to prompt context for LLM generation."""
        return f"""
DESIGN SYSTEM (MANDATORY — every value must come from this system):

BRAND:
  Name: {self.brand_name or "Client"}
  Industry: {self.industry or "General"}
  Tone: {self.tone.value}
  Style: {self.style_description}

TYPOGRAPHY:
  Heading Font: {self.font_heading}
  Body Font: {self.font_body}
  Scale: xs={self.typography.xs}, sm={self.typography.sm}, base={self.typography.base}, lg={self.typography.lg}, xl={self.typography.xl}

COLORS:
  Primary: {self.colors.primary}
  Accent: {self.colors.accent}
  Surface: {self.colors.surface}
  Surface Alt: {self.colors.surface_alt}
  Text Primary: {self.colors.text_primary}
  Text Secondary: {self.colors.text_secondary}
  Border: {self.colors.border}
  Success: {self.colors.success}
  Error: {self.colors.error}

SPACING:
  Unit: {self.spacing.unit}
  Scale: {', '.join(self.spacing.scale[:8])}

LAYOUT:
  Max Width: {self.layout.max_width}
  Columns: {self.layout.columns}
  Gutter: {self.layout.gutter}
  Breakpoints: sm={self.layout.breakpoints['sm']}, md={self.layout.breakpoints['md']}, lg={self.layout.breakpoints['lg']}

COMPONENTS:
  Border Radius: sm={self.border_radius.sm}, md={self.border_radius.md}, lg={self.border_radius.lg}, full={self.border_radius.full}
  Shadows: sm={self.shadow.sm}, md={self.shadow.md}, lg={self.shadow.lg}
  Animation: {self.animation.duration}, easing={self.animation.easing}

ACCESSIBILITY (REQUIRED):
  Minimum Contrast Ratio: {self.accessibility.min_contrast_ratio}:1
  Focus Rings: {'Enabled' if self.accessibility.focus_ring else 'Disabled'}
  Reduced Motion Support: {'Yes' if self.accessibility.reduced_motion_support else 'No'}
  Semantic HTML: {'Required' if self.accessibility.semantic_html else 'Optional'}

RULES:
1. Use ONLY colors from the design system. No arbitrary hex values.
2. Use ONLY fonts from the typography section.
3. All spacing must use the spacing scale values.
4. Every interactive element must have focus and hover states.
5. All images must have alt text. Use semantic HTML.
6. Animations must respect prefers-reduced-motion.
7. Maintain contrast ratio minimum {self.accessibility.min_contrast_ratio}:1.
8. Mobile-first responsive design.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Component Registry Models
# ═══════════════════════════════════════════════════════════════════════════════


class ComponentSource(str, Enum):
    """Component library sources."""

    TWENTYFIRST = "twentyfirst"
    SHADCN = "shadcn"
    ACETERNITY = "aceternity"
    MAGIC_UI = "magicui"
    INTERNAL = "internal"
    CUSTOM = "custom"


@dataclass
class ComponentSpec:
    """Specification for a UI component."""

    name: str
    source: ComponentSource
    category: str  # "hero", "features", "pricing", etc.
    variant: str = "default"
    layout_type: str = "flex"  # "flex", "grid", "stack"
    primary_color: str | None = None
    animation_style: str | None = None
    has_aria_labels: bool = True
    keyboard_navigable: bool = True
    responsive: bool = True
    dark_mode_support: bool = False
    file_path: str | None = None
    prompt_reference: str | None = None

    def compatibility_score(self, design_system: DesignSystem) -> float:
        """Score how well this component matches the design system."""
        score = 0.0

        # Color proximity (0.25 points)
        if self.primary_color and self.primary_color == design_system.colors.primary:
            score += 0.25
        elif self.primary_color:
            # Simple hex distance check
            distance = self._color_distance(self.primary_color, design_system.colors.primary)
            if distance < 0.3:
                score += 0.25 * (1 - distance)

        # Layout compatibility (0.2 points)
        if self.layout_type in ("grid", "flex") and design_system.layout.columns == 12:
            score += 0.2

        # Animation style match (0.2 points)
        if self.animation_style == design_system.tone.value:
            score += 0.2

        # Accessibility compliance (0.2 points)
        if self.has_aria_labels and self.keyboard_navigable:
            score += 0.2

        # Responsive readiness (0.15 points)
        if self.responsive:
            score += 0.15

        # Dark mode support bonus (0.1 points)
        if self.dark_mode_support:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _color_distance(color1: str, color2: str) -> float:
        """Calculate normalized distance between two hex colors (0-1)."""

        def hex_to_rgb(hex_color: str) -> tuple:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        try:
            rgb1 = hex_to_rgb(color1)
            rgb2 = hex_to_rgb(color2)

            # Euclidean distance normalized to 0-1
            distance = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2, strict=False)) ** 0.5
            max_distance = (255**2 * 3) ** 0.5
            return distance / max_distance
        except (ValueError, IndexError):
            return 1.0  # Maximum distance if parsing fails


@dataclass
class ContentBrief:
    """Content brief generated from research."""

    headlines: dict[str, str] = field(default_factory=dict)
    ctas: dict[str, str] = field(default_factory=dict)
    faqs: list[dict[str, str]] = field(default_factory=list)
    testimonials_angles: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    competitor_insights: list[str] = field(default_factory=list)

    def get_section_content(self, section: str) -> dict:
        """Get content for a specific section."""
        return {
            "headline": self.headlines.get(section, ""),
            "cta": self.ctas.get(section, ""),
            "faqs": [f for f in self.faqs if f.get("section") == section],
            "pain_points": self.pain_points[:3] if section == "hero" else [],
        }


@dataclass
class QualityCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    score: float = 1.0
    details: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Overall quality report for generated website."""

    checks: list[QualityCheck] = field(default_factory=list)
    passed: bool = True
    score: float = 0.0
    lighthouse_score: int | None = None
    wcag_level: str = "A"
    responsive_breakpoints_tested: int = 0
    seo_score: int = 0

    def __post_init__(self):
        if self.checks:
            self.passed = all(c.passed for c in self.checks)
            self.score = sum(c.score for c in self.checks) / len(self.checks)
