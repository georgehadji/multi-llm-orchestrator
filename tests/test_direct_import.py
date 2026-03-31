#!/usr/bin/env python3
"""Test direct import to see actual error"""

import sys

sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

print("Testing direct import from dashboard_mission_control...")

try:
    from orchestrator.dashboard_mission_control import run_mission_control

    print(f"✅ SUCCESS: {run_mission_control}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
