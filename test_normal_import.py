#!/usr/bin/env python3
"""Test normal import with debugging."""

import sys

# Monkey-patch the import system to see what's happening
original_import = __builtins__.__import__

counter = 0

def debug_import(name, *args, **kwargs):
    global counter
    counter += 1
    if counter > 1000:
        print(f"WARNING: {counter} imports - might be infinite loop!", file=sys.stderr, flush=True)
    if "code_validator" in name or "assembler" in name:
        print(f"  [{counter}] Importing: {name}", file=sys.stderr, flush=True)
        sys.stderr.flush()
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = debug_import

print("Starting import of orchestrator.code_validator", file=sys.stderr, flush=True)
sys.stderr.flush()

try:
    import orchestrator.code_validator
    print("✓ Import succeeded", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Import failed: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
