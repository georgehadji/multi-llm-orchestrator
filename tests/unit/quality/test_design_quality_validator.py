"""
Unit tests for validate_design_quality validator.

NOTE: validate_design_quality was removed from orchestrator.quality.validators
during a refactoring. These tests are preserved for when the validator
is re-implemented, but are skipped for now.
"""

import pytest

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.skip(
    reason="validate_design_quality removed during refactoring — re-implement when needed"
)


_CLEAN_CSS = """
/* Hallmark · macrostructure: bento_grid · theme: lumen · nav: N1 · footer: Ft1 */
html, body { overflow-x: clip; }
.btn:focus-visible { outline: 2px solid var(--color-accent); }
.btn:active { transform: scale(0.98); }
h1 { font-family: 'Space Grotesk', sans-serif; font-weight: 400; }
.hero { grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); }
"""


@pytest.mark.unit
class TestValidateDesignQuality:
    def test_passes_clean_output(self):
        result = validate_design_quality(_CLEAN_CSS)
        assert result.passed
        assert result.validator_name == "design_quality"

    def test_fails_banned_font(self):
        output = _CLEAN_CSS + "\nfont-family: 'Inter', sans-serif;"
        result = validate_design_quality(output)
        assert not result.passed
        assert "critical" in result.details.lower()

    def test_fails_missing_stamp(self):
        output = "html, body { overflow-x: clip; }\n.btn:focus-visible { outline: 2px solid var(--color-accent); }\n.btn:active { transform: scale(0.98); }"
        result = validate_design_quality(output)
        assert not result.passed
        assert "major" in result.details.lower() or "critical" in result.details.lower()

    def test_registered_in_validators_dict(self):
        assert "design_quality" in VALIDATORS
        assert VALIDATORS["design_quality"] is validate_design_quality

    def test_accepts_genre_override(self):
        output = _CLEAN_CSS + "\nfont-family: 'Inter', sans-serif;"
        # Without genre override: should fail (Inter is banned)
        result = validate_design_quality(output)
        assert not result.passed
