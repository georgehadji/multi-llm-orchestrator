"""
Derive a complete, accessible palette from one brand colour.

A factory cannot ask each client for eleven colours, and it must not let a
client's brand hue produce unreadable text. Given one accent, the palette is
derived by colour theory (hue relationships on the colour wheel, a tonal ramp
for surfaces) and then CHECKED against WCAG contrast — the derivation is
adjusted until it passes, rather than hoped over.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def pal():
    from orchestrator.generators import website_palette

    return website_palette


# ── Contrast maths (the part everything else is checked against) ─────────────


@pytest.mark.unit
class TestContrast:
    def test_black_on_white_is_21_to_1(self, pal):
        assert pal.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.05)

    def test_identical_colours_are_1_to_1(self, pal):
        assert pal.contrast_ratio("#4a7d2c", "#4a7d2c") == pytest.approx(1.0, abs=0.001)

    def test_is_symmetric(self, pal):
        a, b = "#123456", "#fedcba"
        assert pal.contrast_ratio(a, b) == pytest.approx(pal.contrast_ratio(b, a))

    def test_known_pair_matches_wcag_reference(self, pal):
        # #767676 on white is the canonical 4.54:1 boundary case for AA text.
        assert pal.contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.05)

    @pytest.mark.parametrize("bad", ["", "#12", "nope", "#12345g"])
    def test_rejects_malformed_hex(self, pal, bad):
        with pytest.raises(ValueError):
            pal.contrast_ratio(bad, "#ffffff")

    def test_accepts_shorthand_hex(self, pal):
        assert pal.contrast_ratio("#000", "#fff") == pytest.approx(21.0, abs=0.05)


# ── Derivation ───────────────────────────────────────────────────────────────

_ACCENTS = ["#0f5d4a", "#7a3b8f", "#b4451f", "#1d4ed8", "#facc15", "#111111", "#f5f5f5"]


@pytest.mark.unit
class TestDerivePalette:
    @pytest.mark.parametrize("accent", _ACCENTS)
    def test_body_text_is_readable_on_the_page(self, pal, accent):
        p = pal.derive_palette(accent)
        assert pal.contrast_ratio(p["ink"], p["paper"]) >= 7.0, (
            f"body text fails AAA on {accent}: " f"{pal.contrast_ratio(p['ink'], p['paper']):.2f}:1"
        )

    @pytest.mark.parametrize("accent", _ACCENTS)
    def test_button_label_is_readable_on_the_accent(self, pal, accent):
        p = pal.derive_palette(accent)
        assert pal.contrast_ratio(p["accent_ink"], p["accent"]) >= 4.5, (
            f"button text fails AA on {accent}: "
            f"{pal.contrast_ratio(p['accent_ink'], p['accent']):.2f}:1"
        )

    @pytest.mark.parametrize("accent", _ACCENTS)
    def test_secondary_text_still_meets_aa(self, pal, accent):
        p = pal.derive_palette(accent)
        assert pal.contrast_ratio(p["ink_soft"], p["paper"]) >= 4.5

    @pytest.mark.parametrize("accent", _ACCENTS)
    def test_link_colour_is_readable_on_the_page(self, pal, accent):
        p = pal.derive_palette(accent)
        assert (
            pal.contrast_ratio(p["link"], p["paper"]) >= 4.5
        ), f"links fail AA on {accent}: {pal.contrast_ratio(p['link'], p['paper']):.2f}:1"

    def test_returns_every_token_the_stylesheet_uses(self, pal):
        p = pal.derive_palette("#0f5d4a")
        for token in (
            "accent",
            "accent_ink",
            "accent_soft",
            "secondary",
            "ink",
            "ink_soft",
            "paper",
            "surface",
            "line",
            "link",
        ):
            assert token in p, f"missing token: {token}"
            assert p[token].startswith("#") and len(p[token]) == 7

    def test_accent_is_preserved_when_it_is_already_usable(self, pal):
        assert pal.derive_palette("#0f5d4a")["accent"].lower() == "#0f5d4a"

    def test_schemes_place_the_secondary_where_theory_says(self, pal):
        """Complementary sits opposite; analogous sits nearby."""
        comp = pal.derive_palette("#0f5d4a", scheme="complementary")
        ana = pal.derive_palette("#0f5d4a", scheme="analogous")
        base = pal.hue_of("#0f5d4a")
        assert 150 <= pal.hue_distance(base, pal.hue_of(comp["secondary"])) <= 180
        assert pal.hue_distance(base, pal.hue_of(ana["secondary"])) <= 60

    def test_unknown_scheme_is_rejected(self, pal):
        with pytest.raises(ValueError, match="scheme"):
            pal.derive_palette("#0f5d4a", scheme="vibes")

    def test_is_deterministic(self, pal):
        assert pal.derive_palette("#b4451f") == pal.derive_palette("#b4451f")

    def test_surfaces_are_distinguishable_from_the_page(self, pal):
        p = pal.derive_palette("#0f5d4a")
        assert p["surface"] != p["paper"] or p["line"] != p["paper"]

    def test_renders_as_css_custom_properties(self, pal):
        css = pal.palette_to_css(pal.derive_palette("#7a3b8f"))
        assert css.strip().startswith(":root")
        assert "--accent:" in css and "--accent-ink:" in css
        assert css.count(";") >= 10
