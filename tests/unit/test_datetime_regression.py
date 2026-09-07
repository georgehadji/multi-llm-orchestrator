"""Guard: no `datetime.timezone` attribute access where `datetime` is the class.

Why this exists
----------------
`58c1013e` ran a search-and-replace that turned `datetime.utcnow()` into
`datetime.now(datetime.timezone.utc)()` across the codebase. In every module
that writes `from datetime import datetime`, `datetime` at that point is the
*class*, not the module — `datetime.timezone` does not exist on it, so the
expression raises `AttributeError` before the trailing `()` (calling a
`datetime` instance) ever runs. `infrastructure/streaming.py`,
`engine_core/health.py`, `state_mgmt/session_watcher.py` and
`state_mgmt/capability_logger.py` all carried this bug; most sites wrapped it
in a broad `except Exception` that swallowed the crash silently, so it went
unnoticed for months (the same commit produced the dead pickle cache removed
in T5).

A plain regex guard would false-positive on modules that do `import datetime`
(there `datetime.timezone.utc` is correct) — this walks the AST and only
flags the attribute access in modules where `datetime` is bound to the class
via `from datetime import datetime`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "orchestrator"


def _imports_datetime_as_class(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            if any(alias.name == "datetime" and alias.asname is None for alias in node.names):
                return True
    return False


def _find_datetime_timezone_attrs(tree: ast.Module) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "timezone"
            and isinstance(node.value, ast.Name)
            and node.value.id == "datetime"
        ):
            lines.append(node.lineno)
    return lines


def _offending_sites() -> list[str]:
    offenders = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        if "graphify-out" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        if not _imports_datetime_as_class(tree):
            continue
        for lineno in _find_datetime_timezone_attrs(tree):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    return offenders


def test_no_datetime_timezone_attribute_when_datetime_is_the_class():
    offenders = _offending_sites()
    assert not offenders, (
        f"`datetime.timezone` used where `datetime` is the class (from "
        f"`from datetime import datetime`), which raises AttributeError: "
        f"{offenders}. Import `timezone` alongside `datetime` and use "
        f"`timezone.utc` directly."
    )


def test_guard_detects_a_planted_regression(tmp_path):
    planted = tmp_path / "zz_probe.py"
    planted.write_text(
        "from datetime import datetime\n" "x = datetime.now(datetime.timezone.utc)\n",
        encoding="utf-8",
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"), filename=str(planted))
    assert _imports_datetime_as_class(tree)
    assert _find_datetime_timezone_attrs(tree) == [2]
