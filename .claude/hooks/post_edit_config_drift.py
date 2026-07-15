#!/usr/bin/env python3
"""PostToolUse hook: run the config-drift checker after edits to model configs.

Guards the #1 recurring failure mode in this repo: config JSON keys drifting
from Model enum values, silently dropping routing/cost/fallback entries
(see .claude/skills/orchestrator-config-and-flags and the incident chronicle
in orchestrator-failure-archaeology). Fires on edits to orchestrator/config/*.json
or orchestrator/models.py and reports live drift. Advisory only — never blocks.
"""
import json
import subprocess
import sys
from pathlib import Path

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
normalized = path.replace("\\", "/")

watched = "orchestrator/config/" in normalized and normalized.endswith(".json")
watched = watched or normalized.endswith("orchestrator/models.py")
if not watched:
    sys.exit(0)

script = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "orchestrator-diagnostics-and-tooling"
    / "scripts"
    / "check_config_drift.py"
)
if not script.exists():
    sys.exit(0)

result = subprocess.run(
    [sys.executable, str(script)],
    capture_output=True,
    text=True,
    timeout=60,
)

if result.returncode != 0:
    summary = (result.stdout or result.stderr).strip()
    # Keep the tail — the script prints findings last.
    lines = summary.splitlines()
    tail = "\n".join(lines[-15:])
    print(
        "[config-drift] DRIFT DETECTED after this edit — config keys must equal "
        "Model enum values or they silently drop (never alias-resolve around this):\n"
        f"{tail}\n"
        "[config-drift] Fix before committing. See skill: orchestrator-config-and-flags.",
        file=sys.stderr,
    )
else:
    print("[config-drift] OK: enum and config JSONs consistent")

sys.exit(0)  # advisory — the CI gate is the enforcement point
