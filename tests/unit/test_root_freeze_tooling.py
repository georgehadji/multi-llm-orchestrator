"""
Guard for scripts/check_root_module_freeze.py's --update self-rewrite.

The bug: _update_baseline() located the block to replace with
content.find(marker), where `marker` was a string literal that ALSO appeared
verbatim in the function's own source. find() therefore matched the marker
assignment inside the function rather than the baseline block, and the script
rewrote itself starting from the middle of its own code — producing a
SyntaxError and a truncated baseline.

That is why the committed baseline listed 211 modules while 257 root modules
already existed at commit f4429be, the very commit that generated it: the gate
then failed on 45 pre-existing files and stayed red for weeks, catching nothing.

These tests pin the two properties that make the rewrite safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_root_module_freeze.py"
# Assembled, not literal — a literal here is harmless in a test, but it keeps
# the intent obvious: these strings must be unique inside the script.
_BEGIN = "# >>>GENERATED-BASELINE" + "-BEGIN<<<"
_END = "# >>>GENERATED-BASELINE" + "-END<<<"


@pytest.mark.unit
def test_script_is_valid_python():
    ast.parse(_SCRIPT.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_baseline_sentinels_appear_exactly_once():
    """If a sentinel appears twice, --update can rewrite the wrong region."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert src.count(_BEGIN) == 1, f"{_BEGIN} must appear exactly once"
    assert src.count(_END) == 1, f"{_END} must appear exactly once"


@pytest.mark.unit
def test_begin_sentinel_precedes_end_sentinel():
    src = _SCRIPT.read_text(encoding="utf-8")
    assert src.index(_BEGIN) < src.index(_END)


@pytest.mark.unit
def test_baseline_matches_the_actual_root_modules():
    """The committed baseline must equal what is on disk — the whole point."""
    import re

    src = _SCRIPT.read_text(encoding="utf-8")
    block = src[src.index(_BEGIN) : src.index(_END)]
    baseline = set(re.findall(r'"([^"]+\.py)"', block))

    root = Path(__file__).resolve().parents[2] / "orchestrator"
    actual = {p.name for p in root.glob("*.py") if p.name != "__init__.py"}

    assert baseline == actual, (
        f"baseline drifted from disk: only-in-baseline={sorted(baseline - actual)}, "
        f"only-on-disk={sorted(actual - baseline)}"
    )
