#!/usr/bin/env python3
"""
Fail-open silent-failure check.

A *fail-open silent handler* is a broad ``except Exception``/bare ``except``
that neither logs nor re-raises, and returns an affirmative value — ``True``, or
a result object constructed with ``passed=True`` / ``success=True`` / ``ok=True``
/ ``valid=True`` / ``healthy=True`` / ``available=True``.

That combination means a check reports success **because it failed**: the
scan that could not read a file says "clean", the probe that could not connect
says "healthy". It is the highest-severity variant of the silent-failure
pattern, and it produced confirmed defects in hunt tiers T6, T8, T9, T13 and
T16 — each time a validator or scanner reporting a pass it had not earned.

Hunt T18 swept all broad handlers in ``orchestrator/`` and found **zero**
remaining instances of this shape. This check keeps that number at zero.

Note what this does NOT flag, deliberately:

* Handlers that log or re-raise — the failure is surfaced.
* Handlers returning a *negative* result (``passed=False``, ``None``, ``0.0``).
  Failing closed is the safe direction; T18 found several and treated them as
  at most a reporting-quality issue, not a security one.
* Best-effort cleanup (``except Exception: pass`` around a rollback or a
  container teardown). T18 confirmed these are correct: a cleanup that fails
  must not mask the original exception.

Usage::

    python scripts/check_silent_failure.py          # check (CI)
    python scripts/check_silent_failure.py --list   # show what was scanned
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHESTRATOR_DIR = os.path.join(PROJECT_ROOT, "orchestrator")

# Keyword arguments that mean "this check passed".
AFFIRMATIVE_FIELDS = frozenset(
    {"passed", "success", "ok", "valid", "is_valid", "healthy", "available"}
)

# Calls that count as surfacing the failure.
_SURFACING_CALLS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "print"}
)


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """True for `except:` and `except Exception:` / `except BaseException:`."""
    node = handler.type
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in ("Exception", "BaseException")
    if isinstance(node, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id in ("Exception", "BaseException")
            for elt in node.elts
        )
    return False


def _surfaces_failure(handler: ast.ExceptHandler) -> bool:
    """True when the handler logs, prints, or re-raises."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else "")
            )
            if name in _SURFACING_CALLS:
                return True
    return False


def _affirmative_returns(handler: ast.ExceptHandler) -> list[tuple[int, str]]:
    """Return (lineno, description) for each affirmative return in the handler."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(handler):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and value.value is True:
            found.append((node.lineno, "return True"))
        elif isinstance(value, ast.Call):
            for kw in value.keywords:
                if (
                    kw.arg in AFFIRMATIVE_FIELDS
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    func = value.func
                    cls = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "?")
                    found.append((node.lineno, f"return {cls}({kw.arg}=True)"))
    return found


def scan() -> tuple[list[str], int]:
    """Return (violations, number of broad handlers examined)."""
    violations: list[str] = []
    examined = 0
    for dirpath, _dirnames, filenames in os.walk(ORCHESTRATOR_DIR):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except (SyntaxError, OSError):
                continue
            rel = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                    continue
                examined += 1
                if _surfaces_failure(node):
                    continue
                for lineno, what in _affirmative_returns(node):
                    violations.append(f"{rel}:{lineno}: silent handler reports success ({what})")
    return violations, examined


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-open silent-failure check.")
    parser.add_argument("--list", action="store_true", help="print the scan summary")
    args = parser.parse_args()

    violations, examined = scan()

    if args.list:
        print(f"broad exception handlers examined: {examined}")
        print(f"fail-open violations: {len(violations)}")
        return 0

    if violations:
        print("FAIL: silent exception handlers that report success:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            "\nA handler that neither logs nor re-raises must not return an affirmative\n"
            "result — that reports a pass the check did not earn. Log the failure and\n"
            "return a negative result, or let the exception propagate.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {examined} broad handlers examined, none report success on failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
