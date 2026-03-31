#!/usr/bin/env python3
"""Debug script to test direct import."""

import sys
import traceback

print("Step 1: Importing ast", file=sys.stderr, flush=True)
import ast
print("  ✓", file=sys.stderr, flush=True)

print("Step 2: Importing logging", file=sys.stderr, flush=True)
import logging
print("  ✓", file=sys.stderr, flush=True)

print("Step 3: Importing re", file=sys.stderr, flush=True)
import re
print("  ✓", file=sys.stderr, flush=True)

print("Step 4: Importing dataclasses", file=sys.stderr, flush=True)
from dataclasses import dataclass, field
print("  ✓", file=sys.stderr, flush=True)

print("Step 5: Opening code_validator.py", file=sys.stderr, flush=True)
with open("orchestrator/code_validator.py", "r") as f:
    source = f.read()
print(f"  ✓ Read {len(source)} bytes", file=sys.stderr, flush=True)

print("Step 6: Compiling code_validator.py", file=sys.stderr, flush=True)
try:
    code_obj = compile(source, "orchestrator/code_validator.py", "exec")
    print("  ✓", file=sys.stderr, flush=True)
except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

print("Step 7: Creating module namespace", file=sys.stderr, flush=True)
module_ns = {
    "__name__": "orchestrator.code_validator",
    "__file__": "orchestrator/code_validator.py",
    "ast": ast,
    "logging": logging,
    "re": re,
    "dataclass": dataclass,
    "field": field,
}
print("  ✓", file=sys.stderr, flush=True)

print("Step 8: Executing code in namespace", file=sys.stderr, flush=True)
try:
    exec(code_obj, module_ns)
    print("  ✓", file=sys.stderr, flush=True)
except Exception as e:
    print(f"  ✗ Execution error: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("\n✅ All steps completed", file=sys.stderr, flush=True)
