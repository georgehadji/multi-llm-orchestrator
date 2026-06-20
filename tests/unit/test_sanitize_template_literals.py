"""Regression tests for WebsiteGenerator._sanitize_output template-literal handling.

A template literal that mixes a simple interpolation with a complex one (e.g.
`${prefix}-${cond ? 'a' : 'b'}`) must NOT be partially rewritten into string
concatenation — doing so produced invalid TSX like `'prefix' + '-' + ${cond...}`.
Such literals must be left untouched. All-simple literals may still be converted.
"""

import pytest

from orchestrator.generators.website_generator import WebsiteGenerator


@pytest.mark.unit
def test_mixed_template_literal_left_untouched():
    src = "const label = `${prefix}-${cond ? 'a' : 'b'}`;"
    cleaned, _warnings = WebsiteGenerator._sanitize_output(src)
    # The complex ternary interpolation must survive inside the literal,
    # and no corrupt `+ ${...}` concatenation may be produced.
    assert "${cond ? 'a' : 'b'}" in cleaned
    assert "`${prefix}-${cond ? 'a' : 'b'}`" in cleaned
    assert "+ ${" not in cleaned


@pytest.mark.unit
def test_all_simple_template_literal_may_convert():
    src = "const s = `${a}-${b}`;"
    cleaned, _warnings = WebsiteGenerator._sanitize_output(src)
    # No raw ${...} should remain when every interpolation was simple.
    assert "${" not in cleaned
    assert "a" in cleaned and "b" in cleaned


@pytest.mark.unit
def test_plain_template_literal_without_interpolation_unchanged():
    src = "const s = `hello world`;"
    cleaned, _warnings = WebsiteGenerator._sanitize_output(src)
    assert "`hello world`" in cleaned
