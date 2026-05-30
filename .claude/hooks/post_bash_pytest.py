#!/usr/bin/env python3
"""PostToolUse hook: parse pytest output and print a clean summary."""
import json
import re
import sys

data = json.load(sys.stdin)

tool = data.get("tool_name", "")
if tool != "Bash":
    sys.exit(0)

cmd = (data.get("tool_input") or {}).get("command", "")
if "pytest" not in cmd:
    sys.exit(0)

output = (data.get("tool_response") or {}).get("stdout", "")
if not output:
    sys.exit(0)

# Extract the short summary line from pytest output:
# e.g. "===== 5 failed, 42 passed in 12.3s ====="
short = re.search(r"=+ (.+?) =+\s*$", output, re.MULTILINE)
if short:
    print(f"[pytest] {short.group(1).strip()}")

# Print failed test IDs for quick triage (cap at 10)
failed = re.findall(r"FAILED (tests/\S+)", output)
if failed:
    print("[pytest] Failed:")
    for f in failed[:10]:
        print(f"  FAIL {f}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")

# Surface errors (collection errors, import errors)
errors = re.findall(r"ERROR (tests/\S+)", output)
if errors:
    print("[pytest] Errors:")
    for e in errors[:5]:
        print(f"  ERR  {e}")

sys.exit(0)
