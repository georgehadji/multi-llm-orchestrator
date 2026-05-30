#!/usr/bin/env python3
"""PostToolUse hook: run ruff --fix on edited orchestrator Python files."""
import json
import subprocess
import sys

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
if not path.endswith(".py"):
    sys.exit(0)

# Only lint files inside the orchestrator package or tests/
normalized = path.replace("\\", "/")
if "orchestrator/" not in normalized and "tests/" not in normalized:
    sys.exit(0)

result = subprocess.run(
    ["ruff", "check", "--fix", "--quiet", path],
    capture_output=True,
    text=True,
)

filename = path.replace("\\", "/").split("/")[-1]
if result.returncode != 0:
    violations = (result.stdout or result.stderr).strip()
    if violations:
        print(f"[ruff] {filename}: {violations}", file=sys.stderr)
else:
    print(f"[ruff] OK: {filename}")

sys.exit(0)  # never block — linting is advisory at edit time
