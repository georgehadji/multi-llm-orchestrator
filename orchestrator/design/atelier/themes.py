"""
Atelier Theme Registry — 20 catalog themes mapped to DesignSystem tokens.
=======================================================================
Each theme provides an OKLCH palette, font stack, genre classification,
motion direction, and preferred nav/footer archetypes.

Maps directly to:
  - ColorTokens (OKLCH → hex fallback via CSS)
  - TypographyTokens (font pairings)
  - WebsiteConfig.theme selection
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AtelierTheme:
    """A named design theme from the Atelier catalog."""

    slug: str
    name: str
    genre: str  # editorial, modern_minimal, atmospheric, playful, terminal, brutal
    paper_band: str  # light or dark
    paper_oklch: str  # OKLCH background color
    ink_oklch: str  # OKLCH text/foreground color
    accent_oklch: str  # OKLCH accent color
    accent_ink_oklch: str  # OKLCH text-on-accent color
    heading_font: str  # CSS font-family for headings
    body_font: str  # CSS font-family for body text
    display_style: str = ""  # "high-contrast-serif", "geometric-sans", "mono", etc.
    motion_direction: str = ""  # "smooth", "playful", "minimal", "brutal"
    preferred_nav: str = ""  # N1-N13 archetype
    preferred_footer: str = ""  # Ft1-Ft8 archetype
    description: str = ""
    accent_hue: str = ""  # "warm", "cool", "chromatic"


# ── 20 Catalog Themes ─────────────────────────────────────────────────────

ATELIER_THEMES: dict[str, AtelierTheme] = {
    "specimen": AtelierTheme(
        slug="specimen",
        name="Specimen",
        genre="editorial",
        paper_band="light",
        paper_oklch="oklch(0.96 0.002 95)",
        ink_oklch="oklch(0.15 0.01 95)",
        accent_oklch="oklch(0.65 0.22 40)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="smooth",
        preferred_nav="N6",
        preferred_footer="Ft1",
        description="Editorial magazine. High-contrast serif headings.",
        accent_hue="warm orange",
    ),
    "midnight": AtelierTheme(
        slug="midnight",
        name="Midnight",
        genre="modern_minimal",
        paper_band="dark",
        paper_oklch="oklch(0.15 0.01 260)",
        ink_oklch="oklch(0.92 0.005 260)",
        accent_oklch="oklch(0.65 0.18 260)",
        accent_ink_oklch="oklch(0.97 0.005 260)",
        heading_font="Geist",
        body_font="Geist",
        display_style="geometric-sans",
        motion_direction="minimal",
        preferred_nav="N1b",
        preferred_footer="Ft2",
        description="Dark mode. Cool blue. Geometric minimalism.",
        accent_hue="cool blue",
    ),
    "brutal": AtelierTheme(
        slug="brutal",
        name="Brutal",
        genre="brutal",
        paper_band="light",
        paper_oklch="oklch(0.98 0.002 95)",
        ink_oklch="oklch(0.08 0.01 95)",
        accent_oklch="oklch(0.55 0.24 25)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Anton",
        body_font="Geist",
        display_style="condensed-bold",
        motion_direction="brutal",
        preferred_nav="N8",
        preferred_footer="Ft4",
        description="Raw. High-contrast. Condensed bold. Structural honesty.",
        accent_hue="warm red",
    ),
    "garden": AtelierTheme(
        slug="garden",
        name="Garden",
        genre="atmospheric",
        paper_band="light",
        paper_oklch="oklch(0.955 0.008 145)",
        ink_oklch="oklch(0.22 0.03 140)",
        accent_oklch="oklch(0.55 0.18 140)",
        accent_ink_oklch="oklch(0.97 0.005 140)",
        heading_font="Newsreader",
        body_font="Geist",
        display_style="roman-serif",
        motion_direction="smooth",
        preferred_nav="N9",
        preferred_footer="Ft5",
        description="Botanical. Roman serif. Chromatic green. Airy spacing.",
        accent_hue="chromatic-green",
    ),
    "atelier": AtelierTheme(
        slug="atelier",
        name="Atelier",
        genre="editorial",
        paper_band="light",
        paper_oklch="oklch(0.94 0.006 80)",
        ink_oklch="oklch(0.18 0.02 70)",
        accent_oklch="oklch(0.55 0.15 55)",
        accent_ink_oklch="oklch(0.97 0.005 80)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="smooth",
        preferred_nav="N6",
        preferred_footer="Ft1",
        description="Craft workshop. Warm brown. Serif elegance.",
        accent_hue="warm brown",
    ),
    "newsprint": AtelierTheme(
        slug="newsprint",
        name="Newsprint",
        genre="editorial",
        paper_band="light",
        paper_oklch="oklch(0.92 0.004 95)",
        ink_oklch="oklch(0.12 0.015 95)",
        accent_oklch="oklch(0.50 0.20 20)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Fraunces",
        body_font="Newsreader",
        display_style="roman-serif",
        motion_direction="minimal",
        preferred_nav="N1a",
        preferred_footer="Ft1",
        description="Newspaper. Roman serif body. Warm burgundy accent.",
        accent_hue="warm burgundy",
    ),
    "terminal": AtelierTheme(
        slug="terminal",
        name="Terminal",
        genre="terminal",
        paper_band="dark",
        paper_oklch="oklch(0.11 0.01 180)",
        ink_oklch="oklch(0.85 0.04 140)",
        accent_oklch="oklch(0.72 0.20 180)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="JetBrains Mono",
        body_font="Geist Mono",
        display_style="mono",
        motion_direction="minimal",
        preferred_nav="N8",
        preferred_footer="Ft4",
        description="CLI aesthetic. Monospace. Chromatic phosphor. Terminal green.",
        accent_hue="chromatic-phosphor",
    ),
    "manifesto": AtelierTheme(
        slug="manifesto",
        name="Manifesto",
        genre="brutal",
        paper_band="dark",
        paper_oklch="oklch(0.10 0.01 95)",
        ink_oklch="oklch(0.88 0.02 20)",
        accent_oklch="oklch(0.58 0.24 25)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Anton",
        body_font="Geist",
        display_style="condensed-bold",
        motion_direction="brutal",
        preferred_nav="N8",
        preferred_footer="Ft4",
        description="Bold statement. Dark. Red accent. Condensed power.",
        accent_hue="warm red",
    ),
    "almanac": AtelierTheme(
        slug="almanac",
        name="Almanac",
        genre="editorial",
        paper_band="light",
        paper_oklch="oklch(0.94 0.003 250)",
        ink_oklch="oklch(0.20 0.01 250)",
        accent_oklch="oklch(0.50 0.08 250)",
        accent_ink_oklch="oklch(0.97 0.005 250)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="geometric-sans",
        motion_direction="minimal",
        preferred_nav="N1a",
        preferred_footer="Ft2",
        description="Reference work. Cool slate. Clean sans-serif. Trustworthy.",
        accent_hue="cool slate",
    ),
    "sport": AtelierTheme(
        slug="sport",
        name="Sport",
        genre="playful",
        paper_band="light",
        paper_oklch="oklch(0.98 0.002 95)",
        ink_oklch="oklch(0.08 0.01 95)",
        accent_oklch="oklch(0.60 0.22 35)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Anton",
        body_font="Geist",
        display_style="condensed-bold",
        motion_direction="playful",
        preferred_nav="N5",
        preferred_footer="Ft8",
        description="High energy. Condensed bold. Burnt orange. Fast motion.",
        accent_hue="burnt orange",
    ),
    "studio": AtelierTheme(
        slug="studio",
        name="Studio",
        genre="modern_minimal",
        paper_band="light",
        paper_oklch="oklch(0.97 0.003 95)",
        ink_oklch="oklch(0.12 0.01 95)",
        accent_oklch="oklch(0.55 0.16 140)",
        accent_ink_oklch="oklch(0.97 0.005 95)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="smooth",
        preferred_nav="N1b",
        preferred_footer="Ft1",
        description="Design studio. Serif + green accent. Refined minimalism.",
        accent_hue="chromatic-green",
    ),
    "cobalt": AtelierTheme(
        slug="cobalt",
        name="Cobalt",
        genre="modern_minimal",
        paper_band="dark",
        paper_oklch="oklch(0.12 0.015 270)",
        ink_oklch="oklch(0.90 0.005 270)",
        accent_oklch="oklch(0.65 0.20 270)",
        accent_ink_oklch="oklch(0.97 0.005 270)",
        heading_font="Geist",
        body_font="Geist",
        display_style="grotesk-sans",
        motion_direction="minimal",
        preferred_nav="N1b",
        preferred_footer="Ft2",
        description="Deep indigo. Grotesk sans. Electric blue accent. Corporate but bold.",
        accent_hue="cool",
    ),
    "aurora": AtelierTheme(
        slug="aurora",
        name="Aurora",
        genre="atmospheric",
        paper_band="dark",
        paper_oklch="oklch(0.10 0.015 290)",
        ink_oklch="oklch(0.88 0.01 290)",
        accent_oklch="oklch(0.70 0.22 320)",
        accent_ink_oklch="oklch(0.97 0.005 320)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="smooth",
        preferred_nav="N9",
        preferred_footer="Ft5",
        description="Northern lights. Deep purple-violet. Serif elegance. Ethereal.",
        accent_hue="chromatic",
    ),
    "coral": AtelierTheme(
        slug="coral",
        name="Coral",
        genre="playful",
        paper_band="light",
        paper_oklch="oklch(0.95 0.005 20)",
        ink_oklch="oklch(0.20 0.03 20)",
        accent_oklch="oklch(0.65 0.22 25)",
        accent_ink_oklch="oklch(0.97 0.005 20)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="playful",
        preferred_nav="N5",
        preferred_footer="Ft3",
        description="Warm coral reefs. Serif playfulness. Undersea palette.",
        accent_hue="warm",
    ),
    "bloom": AtelierTheme(
        slug="bloom",
        name="Bloom",
        genre="atmospheric",
        paper_band="dark",
        paper_oklch="oklch(0.13 0.01 300)",
        ink_oklch="oklch(0.88 0.02 320)",
        accent_oklch="oklch(0.68 0.22 300)",
        accent_ink_oklch="oklch(0.97 0.005 300)",
        heading_font="Fraunces",
        body_font="Geist",
        display_style="high-contrast-serif",
        motion_direction="smooth",
        preferred_nav="N4",
        preferred_footer="Ft5",
        description="Floral. Deep purple-magenta. Serif. Lush atmosphere.",
        accent_hue="chromatic",
    ),
}


def get_theme(slug: str) -> AtelierTheme | None:
    """Get an Atelier theme by slug name."""
    return ATELIER_THEMES.get(slug.lower())


def list_themes() -> list[str]:
    """Return all available theme slugs."""
    return sorted(ATELIER_THEMES.keys())


def get_themes_by_genre(genre: str) -> list[AtelierTheme]:
    """Filter themes by genre classification."""
    return [t for t in ATELIER_THEMES.values() if t.genre == genre]


def theme_to_prompt_context(theme: AtelierTheme) -> str:
    """Convert an Atelier theme into a prompt injection block.

    Returns a self-contained text block that can be appended to
    the section prompt to guide the LLM's design choices.
    """
    return (
        f"ATELIER DESIGN THEME: {theme.name}\n"
        f"Genre: {theme.genre} | Display: {theme.display_style}\n"
        f"Paper (bg):  {theme.paper_oklch}\n"
        f"Ink (text):  {theme.ink_oklch}\n"
        f"Accent:      {theme.accent_oklch}\n"
        f"Accent Ink:  {theme.accent_ink_oklch}\n"
        f"Heading Font: {theme.heading_font}\n"
        f"Body Font:    {theme.body_font}\n"
        f"Motion:      {theme.motion_direction}\n"
        f"Nav Archetype:  {theme.preferred_nav}\n"
        f"Footer Archetype: {theme.preferred_footer}\n"
        f"Design Philosophy: {theme.description}\n"
        "\n"
        "RULES:\n"
        "1. Use ONLY the OKLCH values above for all backgrounds, text, and accents.\n"
        "2. Use the specified heading and body fonts. No substitutions.\n"
        "3. Match the motion direction: smooth=elegant, playful=bouncy, minimal=subtle, brutal=raw.\n"
        "4. Respect the genre conventions (e.g. editorial = generous whitespace, newspaper layouts).\n"
        "5. Avoid all Hallmark anti-patterns: no purple-cyan gradients, no Inter-only type, no centered heroes.\n"
    )
