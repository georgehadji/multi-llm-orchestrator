"""
Image Generator — creates SVG placeholder images for generated websites.
=======================================================================
Produces self-contained SVG images with design system colors applied.
No external image service, CDN, or API key required.
"""

from pathlib import Path


def generate_images(output_dir: Path, config, design_system) -> None:
    """Generate all placeholder images for a generated website.

    Creates:
      - public/images/og-image.svg — Open Graph social sharing (1200x630)
      - public/images/hero-bg.svg — hero section background with gradient + grid
      - public/images/portfolio-{1,2,3}.svg — work section thumbnails
      - public/images/team-{1,2,3,4}.svg — team photo placeholders (initial + gradient)
      - favicon.svg — site favicon using primary color
      - apple-touch-icon.svg — PWA icon
    """
    img_dir = output_dir / "public" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = design_system
    colors = getattr(ds, "colors", ds)
    primary = getattr(colors, "primary", "#4f9eff")
    accent = getattr(colors, "accent", "#7c3aed")
    bg = getattr(colors, "background", "#09090b")
    surface = getattr(colors, "surface", "#111113")
    text = getattr(colors, "text_primary", "#fafafa")

    site_name = getattr(config, "client_name", "Site") or "Site"
    tagline = getattr(config, "tagline", "") or getattr(
        getattr(config, "content_brief", None), "tagline", ""
    ) or ""

    def _write_svg(filename: str, content: str, parent: Path | None = None):
        (parent or img_dir).mkdir(parents=True, exist_ok=True)
        (parent or img_dir / filename).write_text(content.strip() + "\n", encoding="utf-8")

    # ── OG Image (1200x630) ──
    _write_svg("og-image.svg", svg_og(primary, accent, bg, text, site_name, tagline))

    # ── Hero Background ──
    _write_svg("hero-bg.svg", svg_hero(primary, accent, bg, surface))

    # ── Portfolio Thumbnails (3) ──
    for i, c1, c2 in [(1, primary, accent), (2, accent, primary), (3, surface, primary)]:
        _write_svg(f"portfolio-{i}.svg", svg_portfolio_thumb(i, c1, c2, surface, text))

    # ── Team Avatars (4) ──
    names = ["Alex", "Jordan", "Sam", "Casey"]
    avatar_colors = [primary, accent, "#34d399", "#fbbf24"]
    for i, (name, color) in enumerate(zip(names, avatar_colors), 1):
        _write_svg(f"team-{i}.svg", svg_team_avatar(i, name[0], color, bg, text))

    # ── Favicon + Apple Icon (write to images/ alongside other images) ──
    _write_svg("favicon.svg", svg_icon(32, 6, primary, site_name[0].upper()), img_dir)
    _write_svg(
        "apple-touch-icon.svg", svg_icon(180, 36, bg, site_name[0].upper()), img_dir
    )


def svg_og(primary, accent, bg, text, site_name, tagline):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="og-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{bg}"/>
      <stop offset="50%" style="stop-color:{primary};stop-opacity:0.3"/>
      <stop offset="100%" style="stop-color:{bg}"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#og-grad)"/>
  <text x="600" y="280" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="64" font-weight="700" fill="{text}">{site_name}</text>
  <text x="600" y="350" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="28" fill="{text}" opacity="0.7">{tagline or 'Built with precision'}</text>
  <rect x="500" y="420" width="200" height="4" rx="2" fill="{accent}"/>
</svg>'''


def svg_hero(primary, accent, bg, surface):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900">
  <defs>
    <radialGradient id="hero-grad" cx="50%" cy="40%" r="60%">
      <stop offset="0%" style="stop-color:{accent};stop-opacity:0.15"/>
      <stop offset="100%" style="stop-color:{bg}"/>
    </radialGradient>
    <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
      <path d="M 60 0 L 0 0 0 60" fill="none" stroke="{surface}" stroke-width="0.5" opacity="0.3"/>
    </pattern>
  </defs>
  <rect width="1440" height="900" fill="url(#hero-grad)"/>
  <rect width="1440" height="900" fill="url(#grid)"/>
  <circle cx="1100" cy="200" r="300" fill="{primary}" opacity="0.08"/>
  <circle cx="300" cy="700" r="200" fill="{accent}" opacity="0.06"/>
</svg>'''


def svg_portfolio_thumb(i, c1, c2, surface, text):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400" viewBox="0 0 600 400">
  <defs>
    <linearGradient id="thumb-grad-{i}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{c1};stop-opacity:0.3"/>
      <stop offset="100%" style="stop-color:{c2};stop-opacity:0.1"/>
    </linearGradient>
  </defs>
  <rect width="600" height="400" rx="12" fill="{surface}"/>
  <rect width="600" height="400" rx="12" fill="url(#thumb-grad-{i})"/>
  <text x="300" y="200" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="24" fill="{text}" opacity="0.4">Project {i}</text>
</svg>'''


def svg_team_avatar(i, initial, color, bg, text):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <defs>
    <linearGradient id="avatar-grad-{i}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{color}"/>
      <stop offset="100%" style="stop-color:{bg}"/>
    </linearGradient>
  </defs>
  <circle cx="100" cy="100" r="100" fill="url(#avatar-grad-{i})"/>
  <text x="100" y="115" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="48" font-weight="600" fill="{text}">{initial}</text>
</svg>'''


def svg_icon(size=32, radius=6, bg="#4f9eff", letter="S"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <rect width="{size}" height="{size}" rx="{radius}" fill="{bg}"/>
  <text x="{size // 2}" y="{int(size * 0.7)}" text-anchor="middle" font-family="system-ui,sans-serif"
        font-size="{int(size * 0.55)}" font-weight="700" fill="#fff">{letter}</text>
</svg>'''
