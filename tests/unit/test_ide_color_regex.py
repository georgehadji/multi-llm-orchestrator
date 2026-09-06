"""
Colour-modification cases for the IDE backend's `--accent` swap.

Was `orchestrator/ide_backend/test_color_regex.py`, and broken three ways at
once:

1. It lived inside `orchestrator/`, where `testpaths = ["tests"]` meant pytest
   never collected it. Nothing had run it.
2. It contained no `assert`. It counted failures, printed them, and returned
   `failed == 0`; pytest discards a test's return value, so every case could
   fail and it still reported PASS. Proven by sabotaging the pattern to match
   nothing — still "1 passed".
3. It copy-pasted the production regex instead of calling production, so even
   a green run said nothing about `ide_orchestrator_server.py`.

Once made fail-closed it failed for real: the "No space" case expected
`--accent:#8b5cf6` byte-for-byte, but the substitution always writes
`--accent: <color>`. Both are valid CSS, so the expectation was wrong, not the
code — and the normalisation is now stated in `replace_accent_color`'s
docstring.
"""

from __future__ import annotations

import pytest

from orchestrator.ide_backend.ide_orchestrator_server import replace_accent_color

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "css", "new_color", "expected"),
    [
        ("gold to purple", ":root { --accent: #c9a55c; }", "#8b5cf6", "--accent: #8b5cf6"),
        ("gold to blue", ":root { --accent: #c9a55c; }", "#3b82f6", "--accent: #3b82f6"),
        ("gold to green", ":root { --accent: #c9a55c; }", "#10b981", "--accent: #10b981"),
        ("gold to black", ":root { --accent: #c9a55c; }", "#000000", "--accent: #000000"),
        ("gold to white", ":root { --accent: #c9a55c; }", "#ffffff", "--accent: #ffffff"),
        ("multiple spaces", ":root { --accent:   #c9a55c; }", "#8b5cf6", "--accent: #8b5cf6"),
        # Spacing is normalised, not preserved — both forms are valid CSS.
        ("no space", ":root { --accent:#c9a55c; }", "#8b5cf6", "--accent: #8b5cf6"),
        ("3-digit hex", ":root { --accent: #abc; }", "#8b5cf6", "--accent: #8b5cf6"),
        ("uppercase hex", ":root { --accent: #C9A55C; }", "#8B5CF6", "--accent: #8B5CF6"),
    ],
)
def test_accent_colour_is_replaced(name, css, new_color, expected):
    result, previous = replace_accent_color(css, new_color)

    assert expected in result, f"{name}: expected {expected!r} in {result!r}"
    assert previous is not None, f"{name}: should have reported the replaced declaration"


def test_no_accent_declaration_is_left_untouched():
    """No `--accent` to swap must change nothing and say so."""
    css = ":root { --primary: #0a0a0f; }"
    result, previous = replace_accent_color(css, "#8b5cf6")

    assert result == css
    assert previous is None


def test_only_the_first_accent_is_replaced():
    """The handler passes count=1; a second declaration must survive."""
    css = ":root { --accent: #111111; }\n.dark { --accent: #222222; }"
    result, _ = replace_accent_color(css, "#8b5cf6")

    assert "--accent: #8b5cf6" in result
    assert "--accent: #222222" in result, "the second declaration must be untouched"


def test_round_trip_through_a_css_file(tmp_path):
    """The handler reads, substitutes, writes — keep that path covered.

    This is the one property `test_ide_modifications.py` held that the cases
    above do not; that file was otherwise 9 tests of Python's own `re` module
    plus 3 that imported `SessionManager` and never meaningfully used it.
    """
    css_path = tmp_path / "styles.css"
    css_path.write_text(":root {\n  --primary: #0a0a0f;\n  --accent: #c9a55c;\n}\n")

    updated, previous = replace_accent_color(css_path.read_text(), "#8b5cf6")
    css_path.write_text(updated)

    assert previous == "--accent: #c9a55c"
    on_disk = css_path.read_text()
    assert "--accent: #8b5cf6" in on_disk
    assert "--accent: #c9a55c" not in on_disk
    assert "--primary: #0a0a0f" in on_disk, "unrelated declarations must survive"
