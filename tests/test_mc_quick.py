#!/usr/bin/env python3
"""Quick test"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

try:
    from orchestrator.dashboard_mission_control import run_mission_control
    print("[OK] Import successful!")
    print(f"Function: {run_mission_control}")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
