#!/usr/bin/env python3
"""Final test for Mission Control"""
import sys
import logging

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("="*60)
print("Final Mission Control Test")
print("="*60)

print("\n1. Direct module import...")
try:
    from orchestrator import dashboard_mission_control as dmc
    print(f"✅ Module: {dmc}")
    print(f"✅ Has run_mission_control: {hasattr(dmc, 'run_mission_control')}")
    if hasattr(dmc, 'run_mission_control'):
        print(f"   Value: {dmc.run_mission_control}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Import via orchestrator package...")
try:
    from orchestrator import run_mission_control, MissionControlServer
    print(f"✅ run_mission_control: {run_mission_control}")
    print(f"✅ MissionControlServer: {MissionControlServer}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
