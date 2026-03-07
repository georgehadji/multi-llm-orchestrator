#!/usr/bin/env python3
"""
LLM Orchestrator Standalone Launcher
=====================================
Ξεκινάει το dashboard χωρίς να βασίζεται στο __init__.py
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing dashboard module...")
# Import directly from module
from orchestrator.dashboard_mission_control import run_mission_control
print("Import successful!")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🚀 LLM Orchestrator v6.5.22                                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Starting server... (this may take a few seconds)")
    run_mission_control()
