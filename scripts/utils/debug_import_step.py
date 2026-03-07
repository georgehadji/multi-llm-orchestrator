#!/usr/bin/env python3
"""Debug import step by step"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("Step 1: Importing module...")
try:
    import orchestrator.dashboard_mission_control as dmc
    print(f"✅ Module loaded: {dmc}")
    print(f"   File: {dmc.__file__}")
    
    print("\nStep 2: Checking for run_mission_control...")
    if hasattr(dmc, 'run_mission_control'):
        print(f"✅ Found: {dmc.run_mission_control}")
    else:
        print("❌ NOT FOUND!")
        print(f"   Available: {[x for x in dir(dmc) if not x.startswith('_')]}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
