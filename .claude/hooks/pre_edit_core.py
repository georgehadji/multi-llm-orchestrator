#!/usr/bin/env python3
"""PreToolUse hook: surface architecture rules before editing protected files."""
import json
import sys

RULES = {
    "engine.py": (
        "[architecture-guard] engine.py = MEDIATOR ONLY.\n"
        "  New business logic → new service module. engine.py only wires services.\n"
        "  Run /architecture-guard for the full placement guide."
    ),
    "models.py": (
        "[architecture-guard] models.py = PURE DATA ONLY.\n"
        "  No I/O, no asyncio, no behavior. Only @dataclass and Enum definitions.\n"
        "  Run /architecture-guard for the full placement guide."
    ),
}

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool not in ("Write", "Edit"):
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
filename = path.replace("\\", "/").split("/")[-1]

if filename in RULES:
    print(RULES[filename], file=sys.stderr)

sys.exit(0)  # always allow — informational only
