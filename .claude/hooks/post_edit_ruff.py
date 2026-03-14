"""
PostToolUse hook: run ruff on any orchestrator/*.py file after Edit or Write.
Outputs lint issues as additionalContext so Claude sees them immediately.
"""
import json
import os
import subprocess
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # Malformed input — skip silently

    file_path = data.get("tool_input", {}).get("file_path", "")

    # Only lint Python files inside the orchestrator/ package
    if not file_path.endswith(".py"):
        sys.exit(0)
    if "orchestrator" not in file_path.replace("\\", "/"):
        sys.exit(0)
    if ".claude" in file_path.replace("\\", "/"):
        sys.exit(0)

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-fix", file_path],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )

    if result.returncode == 0:
        sys.exit(0)  # All good — no output needed

    issues = (result.stdout + result.stderr).strip()
    if not issues:
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Ruff lint issues in {os.path.basename(file_path)}:\n{issues}",
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
