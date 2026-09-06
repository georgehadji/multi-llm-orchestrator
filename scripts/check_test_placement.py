#!/usr/bin/env python3
"""
Hunt T21 gate — no uncollected test inside the shipped package.

`pyproject.toml` sets `testpaths = ["tests"]`. A file under `orchestrator/`
that pytest *would* collect by name (`test_*.py` / `*_test.py`) and that
defines module-level test functions or `Test*` classes is therefore a test
that looks real, counts as coverage in a reader's head, and runs nowhere.

T21 found 695 such lines in `ide_backend/`:

  * `test_color_regex.py` — no `assert` at all. It counted failures and
    returned `failed == 0`; pytest discards a test's return value, so it
    reported PASS even with the pattern sabotaged to match nothing. Made
    fail-closed, it failed for real.
  * `test_ide_modifications.py` — 9 tests of Python's own `re` module, plus 3
    written against a different `SessionManager` than the one they imported.
  * `test_server.py` — not a test. A FastAPI server on port 8765, and the file
    `start-ide.bat` actually launched.

The check is deliberately narrow: a *name* alone is not a violation.
`design/slop_test.py` ("Slop Test Engine"), `test_first_generator.py`
(test-first generation), `test_fixer.py` and `test_validator.py` are
production modules whose domain is testing, and they define no tests. Only a
file that both looks collectable and defines tests is flagged.

Run with no arguments; exits 1 on any violation outside BASELINE.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "orchestrator"

# Known offenders, kept out of the failure set so the gate can land green.
# Each entry needs a reason and should shrink, never grow.
BASELINE: dict[str, str] = {
    "orchestrator/test_instructor_tenacity.py": (
        "manual smoke script (`python -m orchestrator.test_instructor_tenacity`) whose "
        "three module-level test functions make LIVE API calls. Renaming it would "
        "change the root-module freeze baseline, so it is recorded rather than moved."
    ),
}


def _is_collectable_name(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py")


def _defines_tests(path: pathlib.Path) -> bool:
    """Module-level `def test_*` / `async def test_*` / `class Test*` only."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return True
    return False


def offenders() -> list[str]:
    return [
        p.relative_to(ROOT).as_posix()
        for p in sorted(PKG.rglob("*.py"))
        if _is_collectable_name(p.name) and _defines_tests(p)
    ]


def main() -> int:
    found = offenders()
    new = [p for p in found if p not in BASELINE]

    if new:
        print(
            "T21 gate: test(s) inside the shipped package that pytest never collects.\n"
            'testpaths = ["tests"], so these run nowhere. Move a real test to '
            "tests/; rename anything that is not a test.\n",
            file=sys.stderr,
        )
        for path in new:
            print(f"  {path}", file=sys.stderr)
        return 1

    stale = [p for p in BASELINE if p not in found]
    if stale:
        print(f"{len(stale)} baseline entr(y/ies) resolved: {', '.join(stale)}")
        print("Remove them from BASELINE in scripts/check_test_placement.py.")
    print(f"T21 gate: OK ({len(found)} baselined, 0 new).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
