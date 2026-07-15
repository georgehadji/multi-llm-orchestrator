#!/usr/bin/env python3
"""PostToolUse hook: catch UTF-8 BOMs in edited Python files immediately.

BOM bytes in .py headers caused Linux-only import failures across 32 files
(commit d913d136); a later audit found new BOM files had crept back in.
Cheaper to catch at write time than in CI. Advisory only.
"""
import json
import sys

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
if not path.endswith(".py"):
    sys.exit(0)

try:
    with open(path, "rb") as f:
        head = f.read(3)
except OSError:
    sys.exit(0)

if head == b"\xef\xbb\xbf":
    filename = path.replace("\\", "/").split("/")[-1]
    print(
        f"[bom-guard] {filename} starts with a UTF-8 BOM — this breaks imports "
        "on Linux/CI (incident d913d136). Rewrite the file without BOM "
        "(encoding='utf-8', not 'utf-8-sig'). Repo-wide check: "
        "python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py",
        file=sys.stderr,
    )

sys.exit(0)  # advisory
