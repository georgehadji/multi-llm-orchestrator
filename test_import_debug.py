#!/usr/bin/env python3
"""Debug import issues"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("Testing direct import from dashboard_mission_control...")

try:
    from orchestrator.dashboard_mission_control import MissionControlServer
    print(f"✅ Direct import successful: {MissionControlServer}")
except Exception as e:
    print(f"❌ Direct import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

try:
    from orchestrator import MissionControlServer
    print(f"✅ Import via __init__ successful: {MissionControlServer}")
except Exception as e:
    print(f"❌ Import via __init__ failed: {e}")
    import traceback
    traceback.print_exc()
