#!/usr/bin/env python3
"""
UTF-8 BOM scanner for the Multi-LLM Orchestrator.

WHY THIS EXISTS
---------------
Commit d913d136 removed UTF-8 BOM bytes (EF BB BF) from 32 .py files. BOMs are
invisible in most Windows editors but cause cryptic SyntaxError/import failures
on Linux CI ("invalid non-printable character U+FEFF"). Windows tools
(PowerShell Out-File, some editors) keep re-introducing them, so this scan
must be cheap to re-run after any bulk file operation on Windows.

WHAT IT CHECKS
--------------
Every .py file under orchestrator/ and tests/ for a leading UTF-8 BOM.
Reads only the first 3 bytes of each file -- fast even on 300+ files.

Usage:
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py --paths orchestrator scripts

Exit codes: 0 = clean, 1 = BOM(s) found, 2 = scan dir missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BOM = b"\xef\xbb\xbf"
DEFAULT_PATHS = ["orchestrator", "tests"]


def scan(paths: list[str]) -> list[Path]:
    """Return .py files under the given repo-relative dirs that start with a BOM."""
    offenders: list[Path] = []
    for rel in paths:
        root = REPO_ROOT / rel
        if not root.exists():
            print(f"ERROR: scan path does not exist: {root}", file=sys.stderr)
            raise SystemExit(2)
        for py in sorted(root.rglob("*.py")):
            with open(py, "rb") as f:
                if f.read(3) == BOM:
                    offenders.append(py)
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help=f"Repo-relative directories to scan (default: {' '.join(DEFAULT_PATHS)})",
    )
    args = parser.parse_args()

    offenders = scan(args.paths)
    scanned = ", ".join(args.paths)

    if offenders:
        print(f"BOM FOUND in {len(offenders)} file(s) under {scanned}:")
        for p in offenders:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print("\nFix (per file, PowerShell):")
        print('  $c = Get-Content -Raw <file>; [IO.File]::WriteAllText("<file>", $c)')
        print("Or (Git Bash): sed -i '1s/^\\xef\\xbb\\xbf//' <file>")
        print("These files WILL break on Linux CI even though they run fine on Windows.")
        return 1

    print(f"OK: no UTF-8 BOM in any .py file under: {scanned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
