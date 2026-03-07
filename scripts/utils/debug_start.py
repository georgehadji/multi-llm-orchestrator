#!/usr/bin/env python3
"""Debug startup"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Importing dashboard module...")
try:
    from orchestrator.dashboard_mission_control import run_mission_control
    print("Step 2: Import successful!")
except Exception as e:
    print(f"ERROR during import: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 3: Starting server...")
try:
    run_mission_control()
except Exception as e:
    print(f"ERROR during startup: {e}")
    traceback.print_exc()
    sys.exit(1)
