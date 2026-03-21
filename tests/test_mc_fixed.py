#!/usr/bin/env python3
"""Test Mission Control after fix"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("Testing imports after fix...")

try:
    from orchestrator.dashboard_mission_control import MissionControlServer
    print(f"[OK] Import successful: {MissionControlServer}")
    
    # Try to create instance
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"[OK] Server created: {server}")
    print(f"[OK] State: {server.state.to_dict()['version']}")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
