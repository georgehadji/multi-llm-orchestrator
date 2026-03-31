#!/usr/bin/env python3
"""Detailed trace of import loop."""

import sys

# Monkey-patch import to track the loop
call_stack = []
original_import = __builtins__.__import__
iterations = 0

def traced_import(name, *args, **kwargs):
    global iterations, call_stack
    iterations += 1

    if iterations > 100:
        print(f"\n!!! INFINITE LOOP DETECTED !!!", file=sys.stderr, flush=True)
        print(f"Call stack (last 10):", file=sys.stderr, flush=True)
        for item in call_stack[-10:]:
            print(f"  {item}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Only trace orchestrator modules
    if "orchestrator" in name or any("orchestrator" in s for s in call_stack):
        call_stack.append(f"[{iterations}] {name}")
        if len(call_stack) > 15:
            call_stack.pop(0)

    return original_import(name, *args, **kwargs)

__builtins__.__import__ = traced_import

print("Starting import of orchestrator package...", file=sys.stderr, flush=True)
try:
    import orchestrator
    print("✓ Success", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Failed: {e}", file=sys.stderr, flush=True)
