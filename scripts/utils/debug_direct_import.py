#!/usr/bin/env python3
"""Debug direct import"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("Attempting direct import...")

try:
    import orchestrator.dashboard_mission_control as dmc
    print(f"✅ Module imported: {dmc}")
    print(f"✅ MissionControlServer: {getattr(dmc, 'MissionControlServer', 'NOT FOUND')}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
