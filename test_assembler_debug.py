#!/usr/bin/env python3
"""Debug script to test assembler import."""

import sys

print("Testing assembler module import", file=sys.stderr, flush=True)
try:
    import orchestrator.assembler
    print("✓ orchestrator.assembler module imported", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Failed to import orchestrator.assembler: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Testing assemble_project function import", file=sys.stderr, flush=True)
try:
    from orchestrator.assembler import assemble_project
    print("✓ assemble_project function imported", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Failed to import assemble_project: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Assembly imports completed", file=sys.stderr, flush=True)
