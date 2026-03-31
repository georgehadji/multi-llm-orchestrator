"""Run test subprocess."""

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "test_frontend_rules.py"],
    capture_output=True,
    text=True,
    cwd=r"E:\Documents\Vibe-Coding\Ai Orchestrator",
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
