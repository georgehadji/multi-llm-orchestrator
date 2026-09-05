"""
Derive a complete, accessible palette from a single brand colour.

A website factory cannot ask every client for eleven colours, and it must not
let a client's brand hue produce unreadable text. So: take one accent, place its
companions by colour theory (hue relationships on the wheel), build a tonal ramp
for surfaces, and then CHECK every foreground/background pair against WCAG
contrast — adjusting lightness until it passes rather than hoping it does.

Targets:
    body text on page      >= 7.0:1   (WCAG AAA)
    secondary text on page >= 4.5:1   (AA)
    link on page           >= 4.5:1   (AA)
    button label on accent >= 4.5:1   (AA)

Stdlib only (``colorsys``). A colour library would be a dependency for maths
that fits on one screen.
"""

from __future__ import annotations

import colorsys
import re

__all__ = [
    "SCHEMES",
    "contrast_ratio",
    "derive_palette",
    "hue_distance",
    "hue_of",
    "palette_to_css",
]

SCHEMES = {
    "analogous": 30.0,  # neighbouring hue — calm, cohesive
    "complementary": 180.0,  # opposite hue — maximum separation
    "triadic": 120.0,  # evenly spaced — lively
}

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _parse(value: str) -> tuple[float, float, float]:
    if not isinstance(value, str) or not _HEX.match(value.strip()):
        raise ValueError(f"not a hex colour: {value!r}")
    raw = value.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    return tuple(int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in rgb)


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG 2.1 relative luminance."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours (1.0 to 21.0)."""
    la, lb = _relative_luminance(_parse(a)), _relative_luminance(_parse(b))
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def hue_of(value: str) -> float:
    """Hue in degrees (0-360)."""
    r, g, b = _parse(value)
    return colorsys.rgb_to_hls(r, g, b)[0] * 360.0


def hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues in degrees (0-180)."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _shift_hue(value: str, degrees: float) -> str:
    h, l, s = colorsys.rgb_to_hls(*_parse(value))
    h = ((h * 360.0 + degrees) % 360.0) / 360.0
    return _hex(colorsys.hls_to_rgb(h, l, s))


def _with_lightness(value: str, lightness: float) -> str:
    h, _, s = colorsys.rgb_to_hls(*_parse(value))
    return _hex(colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s))


def _darken_until(value: str, background: str, target: float) -> str:
    """Walk lightness toward whichever pole gains contrast, until target is met."""
    h, lightness, s = colorsys.rgb_to_hls(*_parse(value))
    bg_light = _relative_luminance(_parse(background)) > 0.5
    step = -0.02 if bg_light else 0.02
    current = value
    for _ in range(50):
        if contrast_ratio(current, background) >= target:
            return current
        lightness = max(0.0, min(1.0, lightness + step))
        current = _hex(colorsys.hls_to_rgb(h, lightness, s))
    # Exhausted the ramp: fall back to the guaranteed pole.
    return "#000000" if bg_light else "#ffffff"


def derive_palette(accent: str, scheme: str = "analogous") -> dict[str, str]:
    """Build a full palette from one accent colour.

    Every returned pair that appears together in the stylesheet is verified to
    meet its WCAG target; see the module docstring for the thresholds.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    _parse(accent)  # validate early

    h, lightness, s = colorsys.rgb_to_hls(*_parse(accent))

    # Surfaces: a near-neutral tint of the brand hue keeps the page feeling
    # related to the brand without competing with it.
    paper = _hex(colorsys.hls_to_rgb(h, 0.985, min(s, 0.18)))
    surface = "#ffffff"
    line = _hex(colorsys.hls_to_rgb(h, 0.90, min(s, 0.15)))

    # Text: start from a very dark tint of the hue, then force it to AAA.
    ink = _darken_until(_hex(colorsys.hls_to_rgb(h, 0.14, min(s, 0.35))), paper, 7.0)
    ink_soft = _darken_until(_hex(colorsys.hls_to_rgb(h, 0.40, min(s, 0.25))), paper, 4.5)

    # Button label: whichever pole reads on the accent as given.
    accent_ink = (
        "#ffffff"
        if contrast_ratio("#ffffff", accent) >= contrast_ratio("#111111", accent)
        else "#111111"
    )
    if contrast_ratio(accent_ink, accent) < 4.5:
        # The brand colour itself cannot carry a label; darken a copy for buttons.
        accent = _darken_until(accent, accent_ink, 4.5)

    # Links must read on the page even when the brand colour is pale.
    link = _darken_until(accent, paper, 4.5)

    return {
        "accent": accent,
        "accent_ink": accent_ink,
        "accent_soft": _with_lightness(accent, min(0.92, lightness + 0.45)),
        "secondary": _shift_hue(accent, SCHEMES[scheme]),
        "ink": ink,
        "ink_soft": ink_soft,
        "paper": paper,
        "surface": surface,
        "line": line,
        "link": link,
    }


def palette_to_css(palette: dict[str, str], selector: str = ":root") -> str:
    """Render a palette as CSS custom properties."""
    body = "".join(f"  --{name.replace('_', '-')}: {value};\n" for name, value in palette.items())
    return f"{selector} {{\n{body}}}\n"
