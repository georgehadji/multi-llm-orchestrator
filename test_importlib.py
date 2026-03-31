#!/usr/bin/env python3
"""Debug script using importlib."""

import sys
import importlib.util

print("Loading code_validator module spec", file=sys.stderr, flush=True)
spec = importlib.util.spec_from_file_location(
    "orchestrator.code_validator",
    "orchestrator/code_validator.py"
)

if spec is None:
    print("ERROR: Could not create module spec", file=sys.stderr, flush=True)
    sys.exit(1)

print("Creating module from spec", file=sys.stderr, flush=True)
module = importlib.util.module_from_spec(spec)

print("Registering module in sys.modules", file=sys.stderr, flush=True)
sys.modules["orchestrator.code_validator"] = module

print("Executing module", file=sys.stderr, flush=True)
try:
    spec.loader.exec_module(module)
    print("✓ Module loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Module execution failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ code_validator module loaded", file=sys.stderr, flush=True)
