"""Reachability census: which orchestrator modules nothing can reach.

Produces the 179-file / 44,449-LOC figure cited in
`docs/plans/2026-09-07-reachability-and-remediation-plan.md` §1.  Run it to
re-derive that number rather than trusting the plan's prose:

    python scripts/reachability_census.py

Classification, since each state calls for a different remedy:

  ENTRYPOINT  - declared in pyproject [project.scripts], or referenced from a
                non-Python file (docs, .bat, CI yaml, Dockerfile). Meant to be
                unimported; NOT dead.
  MAIN-ONLY   - has `if __name__ == "__main__"` but nothing references it.
                Runnable by hand; effectively a script, not a wired module.
  TEST-ONLY   - imported by tests but by no product module. Exercised, unwired.
  SHIM        - re-export shim (tiny, body is imports/`__all__` only).
  ORPHAN      - none of the above. Genuinely unreachable product code.
"""

import ast
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORCH = ROOT / "orchestrator"


def modname(p: Path) -> str:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_by(files):
    refs = defaultdict(set)
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        pkg = modname(f) if f.name == "__init__.py" else modname(f).rsplit(".", 1)[0]
        for n in ast.walk(tree):
            targets = []
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
                    prefix = ".".join(base)
                    mod = f"{prefix}.{n.module}" if n.module else prefix
                else:
                    mod = n.module or ""
                targets = [mod] + [f"{mod}.{a.name}" for a in n.names]
            for t in targets:
                if t.startswith("orchestrator"):
                    refs[t].add(str(f))
    return refs


orch = sorted(ORCH.rglob("*.py"))
tests = sorted((ROOT / "tests").rglob("*.py"))
# scripts/ and the repo-root launchers are product surface too — a module they
# import is reachable even though nothing under orchestrator/ imports it.
extra = [p for p in (ROOT / "scripts").rglob("*.py")] if (ROOT / "scripts").exists() else []
extra += [ROOT / p for p in ("start_dashboard.py",) if (ROOT / p).exists()]
prod_refs = imported_by(orch + extra)
test_refs = imported_by(tests)

# entry points declared in pyproject
pyproject = (ROOT / "pyproject.toml").read_text()
scripts = set(re.findall(r"=\s*\"(orchestrator[\w.]*)", pyproject))

# Dynamic stage discovery: container.py loads these by string, so an AST import
# scan cannot see them.  Both the pyproject entry-point group and the in-code
# fallback list count as real references.
dyn = set(re.findall(r"\"(orchestrator[\w.]+):[A-Za-z_]", pyproject))
dyn |= set(
    re.findall(
        r"\"(orchestrator[\w.]+):[A-Za-z_]",
        (ORCH / "engine_core" / "container.py").read_text(encoding="utf-8", errors="replace"),
    )
)
for d in dyn:
    prod_refs[d].add("<dynamic: entry_points/_FALLBACK_ENTRY_POINTS>")

# any orchestrator.<dotted> mentioned in non-Python files
# Only *operational* non-Python references count as "this is an entry point":
# CI workflows, launcher scripts, packaging, containers.  Excluding docs/ and
# .claude/ is essential — the hunt documentation names nearly every module in
# the repo, which would otherwise mark the entire codebase as an entry point.
try:
    grep = subprocess.run(
        [
            "grep",
            "-rhoE",
            r"orchestrator[a-zA-Z0-9_.]*",
            "--include=*.yml",
            "--include=*.yaml",
            "--include=*.bat",
            "--include=*.toml",
            "--include=*.cfg",
            "--include=Dockerfile*",
            "--include=*.sh",
            "--include=*.ini",
            "--exclude-dir=docs",
            "--exclude-dir=.claude",
            "--exclude-dir=.git",
            "--exclude-dir=node_modules",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    nonpy = set(grep.stdout.split())
except Exception:
    nonpy = set()

buckets = defaultdict(list)
for f in orch:
    if f.name == "__init__.py":
        continue
    m = modname(f)
    src = f.read_text(encoding="utf-8", errors="replace")
    loc = len(src.splitlines())
    if {c for c in prod_refs.get(m, set()) if c != str(f)}:
        continue  # referenced, not dead

    has_main = "__main__" in src
    in_scripts = any(s == m or s.startswith(m + ":") or s.startswith(m + ".") for s in scripts)
    in_nonpy = m in nonpy
    try:
        tree = ast.parse(src)
        is_shim = loc < 60 and all(
            isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)) for n in tree.body
        )
    except SyntaxError:
        is_shim = False

    if in_scripts or in_nonpy:
        b = "ENTRYPOINT"
    elif is_shim:
        b = "SHIM"
    elif has_main:
        b = "MAIN-ONLY"
    elif test_refs.get(m):
        b = "TEST-ONLY"
    else:
        b = "ORPHAN"
    buckets[b].append((loc, str(f.relative_to(ROOT))))

for b in ("ORPHAN", "TEST-ONLY", "MAIN-ONLY", "ENTRYPOINT", "SHIM"):
    rows = sorted(buckets[b], reverse=True)
    print(f"\n### {b}: {len(rows)} files, {sum(r[0] for r in rows)} LOC")
    for loc, p in rows:
        print(f"  {loc:>5}  {p}")
