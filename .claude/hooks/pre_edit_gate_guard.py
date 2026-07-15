#!/usr/bin/env python3
"""PreToolUse hook: surface change-control rules before editing gate files.

Any file that ENFORCES a quality gate is architectural-class per
.claude/skills/orchestrator-change-control — the unwritten rule is
"never weaken a gate to pass": no coverage-floor lowering, no import-linter
exemptions, no root-allowlist expansion, no xfail-to-green without review.
Informational only — legitimate, reviewed gate work must stay possible.
"""
import json
import sys

GATE_PATTERNS = (
    ".importlinter",
    ".github/workflows/",
    ".pre-commit-config.yaml",
    "scripts/check_new_root_files.py",
    "tests/unit/test_preexisting_problems.py",
)

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
normalized = path.replace("\\", "/")

hit = next((p for p in GATE_PATTERNS if p in normalized), None)

# pyproject.toml only matters for its gate sections; warn on any edit but say why.
if hit is None and normalized.endswith("pyproject.toml"):
    hit = "pyproject.toml"

if hit:
    print(
        f"[change-control] {hit} is a GATE file — edits are architectural-class.\n"
        "  Never weaken a gate to pass: no coverage-floor lowering, no new\n"
        "  import-linter exemptions, no KERNEL_ALLOWLIST expansion, no\n"
        "  xfail-to-green. Gates only ratchet tighter without explicit approval.\n"
        "  Full rules + escalation path: skill orchestrator-change-control.",
        file=sys.stderr,
    )

sys.exit(0)  # always allow — review discipline, not a hard lock
