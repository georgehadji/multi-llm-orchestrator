#!/usr/bin/env python3
"""Test runner that writes results to file"""

import subprocess
import sys
import os

# Run quick test
print("Running quick_test.py...")
result = subprocess.run(
    [sys.executable, "quick_test.py"], capture_output=True, text=True, cwd=os.getcwd()
)

# Write results
with open("test_results.txt", "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== Return Code: {result.returncode} ===\n")

print(f"Test completed with return code: {result.returncode}")
print("Results written to test_results.txt")
