#!/usr/bin/env python3
"""Test basic imports"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    print("1. Importing models...")
    from orchestrator.models import Budget
    print("   OK")
    
    print("2. Importing engine...")
    from orchestrator.engine import Orchestrator
    print("   OK")
    
    print("3. Importing telemetry...")
    from orchestrator.telemetry import TelemetryCollector
    print("   OK")
    
    print("4. Importing dashboard...")
    from orchestrator.dashboard_mission_control import run_mission_control
    print("   OK")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
