"""
T0 (census repair) proof-of-defect and no-regression tests.

Two VERIFIED DEFECTs from docs/hunts/t0-census-repair/inventory.md:

C1 — CLAUDE.md asserted, twice, that `tests/stress_test.py` exists with
     documented pre-existing failures (S2, S6, S7). The file was never
     committed (`projects/stress_test/README.md` says so directly). A false
     documented invariant is exactly what a defect-hunt's Phase 3b innocence
     attempt would have accepted at face value to wrongly clear a real defect.

C2 — orchestrator/quality/toml_validator.py did `import tomllib`
     unconditionally. `tomllib` is stdlib only from Python 3.11 onward, but
     pyproject.toml declares `requires-python = ">=3.10"` (and lists the
     3.10 classifier) with no `tomli` backport dependency. Verified directly:
     `/usr/bin/python3.10 -c "import tomllib"` raises ModuleNotFoundError;
     `/usr/bin/python3.11` does not. CI never catches this because every job
     pins Python 3.12.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_claude_md_does_not_assert_stress_test_py_has_known_failures():
    content = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "S2, S6, S7" not in content, (
        "CLAUDE.md re-asserts specific pre-existing failures in a test file "
        "that was never committed — see projects/stress_test/README.md"
    )


@pytest.mark.unit
def test_c1_stress_test_py_still_does_not_exist():
    """If this starts existing, the corrected CLAUDE.md wording ('never
    committed') needs revisiting too — this pins the fact the fix relies on."""
    assert not (REPO_ROOT / "tests" / "stress_test.py").exists()


# --- C2 -------------------------------------------------------------------


@pytest.mark.unit
def test_c2_toml_validator_has_version_gated_tomllib_import():
    """No-regression guard: the unconditional `import tomllib` must not come back."""
    src = (REPO_ROOT / "orchestrator" / "quality" / "toml_validator.py").read_text(encoding="utf-8")
    assert re.search(r"^import tomllib\s*$", src, re.MULTILINE) is None, (
        "unconditional `import tomllib` regressed — this breaks on Python 3.10, "
        "which pyproject.toml's requires-python still declares as supported"
    )
    assert "sys.version_info >= (3, 11)" in src
    assert "import tomli as tomllib" in src


@pytest.mark.unit
def test_c2_pyproject_declares_tomli_for_pre_311():
    src = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'"tomli[^"]*python_version\s*<\s*[\'"]3\.11[\'"]', src), (
        "pyproject.toml must declare a tomli fallback dependency for python_version < 3.11 "
        "to match toml_validator.py's import guard"
    )


@pytest.mark.unit
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="exercises the stdlib tomllib codepath, which only exists on 3.11+; "
    "the <3.11 tomli fallback path is covered by test_c2_toml_validator_has_version_gated_tomllib_import",
)
def test_c2_module_still_imports_and_validates_on_current_interpreter():
    """Boundary: on the interpreter actually running this suite (3.11+), the
    fix must be a no-op — same stdlib tomllib, same validate_toml() behavior."""
    import tomllib as stdlib_tomllib

    from orchestrator.quality import toml_validator

    assert toml_validator.tomllib is stdlib_tomllib

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write('key = "value"\n')
        good_path = Path(f.name)
    try:
        is_valid, error = toml_validator.validate_toml(good_path)
        assert is_valid and error == ""
    finally:
        good_path.unlink()

    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("key = [1, 2\n")  # unclosed array — invalid TOML
        bad_path = Path(f.name)
    try:
        is_valid, error = toml_validator.validate_toml(bad_path)
        assert not is_valid and error
    finally:
        bad_path.unlink()
