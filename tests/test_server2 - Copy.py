#!/usr/bin/env python3
"""Test server startup with skip fix"""

import asyncio
import sys
import os
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "orchestrator"))

from dashboard_mission_control import run_mission_control

print("=" * 50)
print("TESTING SERVER STARTUP (with skip fix)")
print("=" * 50)

if __name__ == "__main__":
    try:
        run_mission_control()
    except KeyboardInterrupt:
        print("Server stopped")
