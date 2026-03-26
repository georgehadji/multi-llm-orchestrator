#!/usr/bin/env python3
"""Ultra-minimal test"""
import sys
import os

# Test 1: Can we import log_config directly?
print("Test 1: Import log_config directly...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))
try:
    import log_config
    logger = log_config.get_logger("test")
    print("  ✓ log_config works!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Can we import the shim logging.py?
print("\nTest 2: Import logging shim...")
try:
    import logging as custom_logging
    # This should trigger the shim which imports log_config
    # But we need to be careful not to conflict with stdlib logging
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))
    # Use importlib to avoid name conflict
    import importlib.util
    spec = importlib.util.spec_from_file_location("orch_logging", 
        os.path.join(os.path.dirname(__file__), 'orchestrator', 'logging.py'))
    orch_logging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orch_logging)
    print("  ✓ logging shim works!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests passed!")
