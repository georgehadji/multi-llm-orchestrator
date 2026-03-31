#!/usr/bin/env python3
"""Test importing models directly without using orchestrator.__init__"""

import sys
import os

# Make sure we're using the worktree directory
sys.path.insert(0, os.getcwd())

print("Attempting direct import of orchestrator/models.py...", flush=True)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("models_direct", "orchestrator/models.py")
    models = importlib.util.module_from_spec(spec)
    sys.modules["models_direct"] = models
    spec.loader.exec_module(models)
    print("✓ models.py loaded directly")
except Exception as e:
    print(f"✗ Failed to load models.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ Success")
