#!/usr/bin/env python3
"""
Environment-flag inventory for the Multi-LLM Orchestrator.

WHY THIS EXISTS
---------------
The repo has been bitten by "dead flags" -- env vars that are declared/read
but never affect the call path (e.g. USE_PROVIDER_SORTING, confirmed dead
2026-06-25). Before trusting any USE_*/ENABLE_*/ORCH_* flag, find where it is
actually READ. A flag read in exactly one config dataclass and never consumed
downstream is a dead-flag suspect.

WHAT IT DOES
------------
Scans .py files under orchestrator/ (and optionally tests/, scripts/) for
os.environ / os.getenv reads of names matching USE_*, ENABLE_*, ORCH_*, and
*_API_KEY, then prints a flag -> [file:line] report. Pure stdlib, regex-based;
it finds READS of literal names, not indirect/dynamic lookups.

Usage:
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py --paths orchestrator tests
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py --flag USE_RESPONSE_HEALING

Exit codes: 0 = report printed, 1 = --flag given and not found anywhere, 2 = bad path.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# Matches os.environ["X"], os.environ.get("X"...), os.getenv("X"...),
# environ.get("X") -- capturing the flag name.
_READ_PATTERN = re.compile(
    r"""(?:os\.environ(?:\.get)?|os\.getenv|environ\.get)\s*[\(\[]\s*["']([A-Z][A-Z0-9_]+)["']"""
)
_FLAG_NAME = re.compile(r"^(USE_|ENABLE_|ORCH_)|_API_KEY$")


def scan(paths: list[str]) -> dict[str, list[str]]:
    """Return {flag_name: ['relpath:lineno', ...]} for all matching env reads."""
    hits: dict[str, list[str]] = defaultdict(list)
    for rel in paths:
        root = REPO_ROOT / rel
        if not root.exists():
            print(f"ERROR: scan path does not exist: {root}", file=sys.stderr)
            raise SystemExit(2)
        for py in sorted(root.rglob("*.py")):
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                for m in _READ_PATTERN.finditer(line):
                    name = m.group(1)
                    if _FLAG_NAME.search(name):
                        hits[name].append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["orchestrator"],
        help="Repo-relative directories to scan (default: orchestrator)",
    )
    parser.add_argument(
        "--flag",
        help="Report only this flag; exit 1 if it is never read (dead-flag check).",
    )
    args = parser.parse_args()

    hits = scan(args.paths)

    if args.flag:
        locs = hits.get(args.flag, [])
        if not locs:
            print(f"NOT READ ANYWHERE under {args.paths}: {args.flag} (dead-flag suspect)")
            return 1
        print(f"{args.flag} — read at {len(locs)} site(s):")
        for loc in locs:
            print(f"  {loc}")
        return 0

    print(f"Env flags (USE_*/ENABLE_*/ORCH_*/*_API_KEY) read under: {', '.join(args.paths)}\n")
    for name in sorted(hits):
        locs = hits[name]
        marker = "  [single read site — verify it is consumed downstream]" if len(locs) == 1 else ""
        print(f"{name}  ({len(locs)} read site(s)){marker}")
        for loc in locs:
            print(f"    {loc}")
    print(f"\nTotal distinct flags: {len(hits)}")
    print("NOTE: a flag being READ does not prove it is WIRED. USE_PROVIDER_SORTING is")
    print("read but confirmed dead in the call path (2026-06-25). Trace consumers before trusting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
