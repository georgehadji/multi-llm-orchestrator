#!/usr/bin/env python3
"""Regenerate docs/hunts/v4_waves.tsv — the V4 precision-audit wave manifest.

Scores every orchestrator/*.py file on the four axes V4 Phase 0.4 names
(exposure, mutation, complexity, churn), then packs the audited tiers into
waves that respect SCAN_BUDGET: 40 files or 15k LOC, whichever binds first.

The scores are PROXIES, not ground truth — regex counts, not semantics. They
decide reading order and tier, never a finding's severity. A file landing in
the un-audited tail is a statement about where budget went, not a claim that
it is defect-free.

Usage:  python scripts/plan_v4_waves.py [--check]
        --check exits 1 if the committed manifest is stale.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs/hunts/v4_waves.tsv"

MAX_FILES, MAX_LOC = 40, 15_000
DEEP_MIN, STD_MIN = 6, 4  # priority bands; below STD_MIN is the declared tail

ENTRY = re.compile(
    r"@app\.(get|post|put|delete)|@router\.|FastAPI\(|argparse|def main\(|@click\.|websocket|def handle_"
)
SINK = re.compile(
    r"subprocess|os\.system|\beval\(|\bexec\(|pickle\.|yaml\.load|\.execute\(|requests\.|httpx\.|open\("
)
MUT = re.compile(
    r"open\([^)]*['\"][wa]|\.execute\(|aiosqlite|sqlite3|\.commit\(|cost|budget|price|\.write\(|os\.remove|shutil\."
)
CPLX = re.compile(r"^\s*(if|for|while|except|elif|async def|with )", re.M)


def churn_counts() -> dict[str, int]:
    """Commits per file on origin/master. Absent history scores 0, not an error."""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "origin/master",
                "-400",
                "--name-only",
                "--pretty=format:",
                "--",
                "orchestrator",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.endswith(".py"):
            counts[line] = counts.get(line, 0) + 1
    return counts


def score_files() -> list[tuple[int, int, str]]:
    churn = churn_counts()
    rows = []
    for f in sorted((ROOT / "orchestrator").rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        src = f.read_text(encoding="utf-8", errors="replace")
        loc = src.count("\n")  # newline count, matching `wc -l`
        exposure = 3 if ENTRY.search(src) else (2 if SINK.search(src) else 0)
        mutation = min(3, len(MUT.findall(src)) // 4)
        complexity = min(3, len(CPLX.findall(src)) // 40)
        c = churn.get(rel, 0)
        rows.append(
            (
                exposure
                + mutation
                + complexity
                + (3 if c >= 8 else 2 if c >= 4 else 1 if c else 0),
                loc,
                rel,
            )
        )
    rows.sort(key=lambda r: (-r[0], -r[1]))
    return rows


def pack(rows):
    """Greedy: close a wave when either budget cap would be exceeded."""
    waves, cur, cur_loc = [], [], 0
    for row in rows:
        if cur and (len(cur) + 1 > MAX_FILES or cur_loc + row[1] > MAX_LOC):
            waves.append(cur)
            cur, cur_loc = [], 0
        cur.append(row)
        cur_loc += row[1]
    if cur:
        waves.append(cur)
    return waves


def render(rows) -> str:
    deep = pack([r for r in rows if r[0] >= DEEP_MIN])
    std = pack([r for r in rows if STD_MIN <= r[0] < DEEP_MIN])
    lines = [
        "# V4 precision-audit wave manifest — generated, do not hand-edit",
        "# regenerate: python scripts/plan_v4_waves.py",
        "#wave\ttier\tpriority\tloc\tpath",
    ]
    n = 0
    for tier, waves in (("DEEP", deep), ("STD", std)):
        for wave in waves:
            n += 1
            lines += [f"P{n}\t{tier}\t{p}\t{loc}\t{path}" for p, loc, path in wave]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the manifest is stale")
    args = ap.parse_args()

    text = render(score_files())
    if args.check:
        if not MANIFEST.exists() or MANIFEST.read_text() != text:
            print("docs/hunts/v4_waves.tsv is stale — run: python scripts/plan_v4_waves.py")
            return 1
        print("v4_waves.tsv up to date")
        return 0
    MANIFEST.write_text(text)
    audited = sum(1 for line in text.splitlines() if line.startswith("P"))
    print(
        f"wrote {MANIFEST.relative_to(ROOT)}: {audited} files across "
        f"{len({line.split(chr(9))[0] for line in text.splitlines() if line.startswith('P')})} waves"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
