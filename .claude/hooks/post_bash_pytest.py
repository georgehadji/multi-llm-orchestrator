"""
PostToolUse hook: after any pytest run, warn if coverage drops below threshold.
Parses the pytest --cov output from the tool response and alerts Claude.
"""
import json
import re
import sys

COVERAGE_THRESHOLD = 70  # percent


def extract_coverage(output: str) -> int | None:
    """Extract the TOTAL coverage percentage from pytest-cov output."""
    # Matches: TOTAL    1234    567    54%
    match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", output, re.MULTILINE)
    if match:
        return int(match.group(1))
    return None


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    command = data.get("tool_input", {}).get("command", "")
    if "pytest" not in command:
        sys.exit(0)

    tool_response = data.get("tool_response", "")
    if not isinstance(tool_response, str):
        sys.exit(0)

    coverage = extract_coverage(tool_response)
    if coverage is None:
        sys.exit(0)  # No coverage data (e.g. --no-cov run)

    if coverage >= COVERAGE_THRESHOLD:
        sys.exit(0)  # Above threshold — no warning needed

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"⚠️  Coverage is {coverage}% — below the {COVERAGE_THRESHOLD}% threshold. "
                f"Add tests to cover the new code before committing."
            ),
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
