#!/usr/bin/env python3
"""Test simplified Mission Control"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("Testing simplified Mission Control...")

try:
    from orchestrator.dashboard_mission_control import MissionControlServer, run_mission_control
    print(f"[OK] Import successful")
    
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"[OK] Server created")
    print(f"[OK] State: {server.state.to_dict()}")

    print("\nReady to start!")
    print("   python start_mission_control.py")

except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
