"""Unit tests for slop_test engine."""

import pytest

from orchestrator.design.slop_test import SlopTestEngine

# A truly clean output that satisfies all structural gates
_CLEAN_OUTPUT = """
/* Hallmark · macrostructure: bento_grid · theme: lumen · nav: N1 · footer: Ft1 */
html, body { overflow-x: clip; }
.btn:focus-visible { outline: 2px solid var(--color-accent); }
.btn:active { transform: scale(0.98); }
h1 { font-family: 'Space Grotesk', sans-serif; font-weight: 400; }
.hero { grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); }
"""


@pytest.mark.unit
class TestSlopTestEngine:
    def test_clean_output_passes(self):
        engine = SlopTestEngine()
        result = engine.run(_CLEAN_OUTPUT)
        assert result.passed, f"Unexpected findings: {result.summary}"
        assert len(result.findings) == 0

    def test_detects_ai_gradient(self):
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nbackground: linear-gradient(135deg, #8b5cf6, #3b82f6);"
        result = engine.run(output)
        assert not result.passed
        assert any("gradient" in f.gate.name.lower() for f in result.findings)

    def test_detects_banned_font_inter(self):
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nfont-family: 'Inter', sans-serif;"
        result = engine.run(output)
        assert not result.passed
        assert any("font" in f.gate.name.lower() for f in result.findings)

    def test_detects_pure_black_bg(self):
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nbackground-color: #000;"
        result = engine.run(output)
        assert not result.passed
        assert any("black" in f.gate.name.lower() for f in result.findings)

    def test_detects_centered_everything(self):
        # Gate 7 regex requires "hero" followed by text-align:center 3+ times
        # without semicolons between (specific format check)
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nhero text-align: center text-align: center text-align: center"
        result = engine.run(output)
        assert not result.passed
        assert any(
            "centred" in f.gate.name.lower() for f in result.findings
        ), f"Findings: {result.summary}"

    def test_detects_inline_color(self):
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\ncolor: #ff5733;"
        result = engine.run(output)
        assert not result.passed
        assert any("token" in f.gate.name.lower() for f in result.critical_failures)

    def test_genre_override_allows_atmospheric_center(self):
        # Gate 7 is overridden for atmospheric genre
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nhero text-align: center text-align: center text-align: center"
        result = engine.run(output, genre="atmospheric")
        assert result.passed, f"Atmospheric should allow centred hero: {result.summary}"

    def test_summary_format(self):
        engine = SlopTestEngine()
        output = _CLEAN_OUTPUT + "\nfont-family: 'Inter', sans-serif; color: #000;"
        result = engine.run(output)
        assert "findings" in result.summary.lower()
        assert len(result.findings) >= 1

    def test_missing_stamp_fails(self):
        engine = SlopTestEngine()
        output = """
        html, body { overflow-x: clip; }
        .btn:focus-visible { outline: 2px solid var(--color-accent); }
        .btn:active { transform: scale(0.98); }
        """
        result = engine.run(output)
        assert not result.passed
        assert any("stamp" in f.gate.name.lower() for f in result.findings)

    def test_missing_interactive_states_fails(self):
        engine = SlopTestEngine()
        output = """
        /* Hallmark · macrostructure: bento_grid · theme: lumen · nav: N1 · footer: Ft1 */
        html, body { overflow-x: clip; }
        """
        result = engine.run(output)
        assert not result.passed
        assert any("interactive" in f.gate.name.lower() for f in result.findings)

    def test_missing_overflow_clip_fails(self):
        engine = SlopTestEngine()
        output = """
        /* Hallmark · macrostructure: bento_grid · theme: lumen · nav: N1 · footer: Ft1 */
        .btn:focus-visible { outline: 2px solid var(--color-accent); }
        .btn:active { transform: scale(0.98); }
        """
        result = engine.run(output)
        assert not result.passed
        assert any("horizontal scroll" in f.gate.name.lower() for f in result.findings)
