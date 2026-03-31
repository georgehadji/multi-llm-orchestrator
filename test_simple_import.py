#!/usr/bin/env python3
"""Test simple cache import."""

import sys
print("Starting...", flush=True)
sys.stdout.flush()

try:
    print("Importing cache", flush=True)
    sys.stdout.flush()
    from orchestrator.cache import DiskCache
    print("Success!")
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
