#!/usr/bin/env python3
"""Debug script to test code_validator import."""

import sys

print("Testing code_validator import", file=sys.stderr, flush=True)
try:
    from orchestrator.code_validator import extract_code_from_llm_response, validate_code
    print("✓ code_validator imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"✓ code_validator not available (expected): {e}", file=sys.stderr, flush=True)
except Exception as e:
    print(f"✗ Failed to import code_validator: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()

print("\n✅ code_validator test completed", file=sys.stderr, flush=True)
