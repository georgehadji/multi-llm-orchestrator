"""Unit tests for validate_anti_slop design validator."""

import pytest


@pytest.mark.unit
def test_anti_slop_flags_inter_font():
    from orchestrator.quality.design_validators import validate_anti_slop

    html = "<style>body { font-family: 'Inter', sans-serif; }</style>"
    result = validate_anti_slop(html)
    assert not result.passed
    assert "font" in result.details.lower() or "inter" in result.details.lower()


@pytest.mark.unit
def test_anti_slop_flags_roboto_font():
    from orchestrator.quality.design_validators import validate_anti_slop

    css = "body { font-family: Roboto, sans-serif; }"
    result = validate_anti_slop(css)
    assert not result.passed


@pytest.mark.unit
def test_anti_slop_flags_pure_black_background():
    from orchestrator.quality.design_validators import validate_anti_slop

    css = ".hero { background: #000000; color: white; }"
    result = validate_anti_slop(css)
    assert not result.passed
    assert "#000" in result.details or "Pure" in result.details or "black" in result.details.lower()


@pytest.mark.unit
def test_anti_slop_flags_pure_white_background():
    from orchestrator.quality.design_validators import validate_anti_slop

    css = "body { background: #fff; }"
    result = validate_anti_slop(css)
    assert not result.passed


@pytest.mark.unit
def test_anti_slop_flags_uniform_three_column_grid():
    from orchestrator.quality.design_validators import validate_anti_slop

    css = ".features { display: grid; grid-template-columns: repeat(3, 1fr); }"
    result = validate_anti_slop(css)
    assert not result.passed
    assert (
        "3" in result.details
        or "column" in result.details.lower()
        or "grid" in result.details.lower()
    )


@pytest.mark.unit
def test_anti_slop_flags_tailwind_grid_cols_3():
    from orchestrator.quality.design_validators import validate_anti_slop

    html = '<div class="grid grid-cols-3 gap-4">'
    result = validate_anti_slop(html)
    assert not result.passed


@pytest.mark.unit
def test_anti_slop_flags_generic_box_shadow():
    from orchestrator.quality.design_validators import validate_anti_slop

    css = ".card { box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }"
    result = validate_anti_slop(css)
    assert not result.passed


@pytest.mark.unit
def test_anti_slop_flags_em_dash():
    from orchestrator.quality.design_validators import validate_anti_slop

    html = "<p>Our product — built for professionals.</p>"
    result = validate_anti_slop(html)
    assert not result.passed
    assert (
        "em" in result.details.lower() or "dash" in result.details.lower() or "—" in result.details
    )


@pytest.mark.unit
def test_anti_slop_passes_clean_intentional_design():
    from orchestrator.quality.design_validators import validate_anti_slop

    clean_html = """
    <style>
      body { font-family: 'Geist', sans-serif; background: #0a0a0a; color: #fafafa; }
      .hero { padding: 6rem 2rem; }
      .features { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; }
      .card { box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15); }
    </style>
    <h1>Build faster</h1>
    <p>Our product is built for professionals.</p>
    """
    result = validate_anti_slop(clean_html)
    assert result.passed, f"Expected clean design to pass, got: {result.details}"


@pytest.mark.unit
def test_anti_slop_validator_name():
    from orchestrator.quality.design_validators import validate_anti_slop

    result = validate_anti_slop("")
    assert result.validator_name == "anti_slop"


@pytest.mark.unit
def test_anti_slop_registered_in_validators_dict():
    from orchestrator.quality.validators import VALIDATORS

    assert "anti_slop" in VALIDATORS
    validator_fn = VALIDATORS["anti_slop"]
    assert callable(validator_fn)
    # Verify it returns a ValidationResult-like object
    result = validator_fn("")
    assert hasattr(result, "passed")
    assert hasattr(result, "validator_name")


@pytest.mark.unit
def test_multiple_findings_reported():
    from orchestrator.quality.design_validators import validate_anti_slop

    bad_html = """
    <style>
      body { font-family: Inter, sans-serif; background: #000; }
      .grid { grid-template-columns: repeat(3, 1fr); }
    </style>
    """
    result = validate_anti_slop(bad_html)
    assert not result.passed
    assert "3 total" in result.details or result.details.count("- ") >= 2
