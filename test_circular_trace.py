#!/usr/bin/env python3
"""Trace circular imports."""

import sys

# Track what's being imported
importing = set()
imported = set()

original_import = __builtins__.__import__

call_stack = []

def debug_import(name, *args, **kwargs):
    global call_stack

    if "code_validator" in name or "output_writer" in name or "assembler" in name:
        indent = "  " * len(call_stack)
        print(f"{indent}→ {name}", file=sys.stderr, flush=True)

        if name in importing:
            # Circular import detected
            print(f"{indent}  ⚠️  CIRCULAR: {' → '.join(importing)} → {name}", file=sys.stderr, flush=True)

        call_stack.append(name)
        importing.add(name)

    try:
        result = original_import(name, *args, **kwargs)
        if "code_validator" in name or "output_writer" in name or "assembler" in name:
            call_stack.pop()
            importing.discard(name)
            imported.add(name)
        return result
    except Exception as e:
        if "code_validator" in name or "output_writer" in name or "assembler" in name:
            call_stack.pop()
            print(f"  ✗ Failed: {e}", file=sys.stderr, flush=True)
        raise

__builtins__.__import__ = debug_import

print("Attempting to import orchestrator.assembler", file=sys.stderr, flush=True)
sys.stderr.flush()

try:
    from orchestrator.assembler import assemble_project
    print("✓ Success", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Failed: {e}", file=sys.stderr, flush=True)
