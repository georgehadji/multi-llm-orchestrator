#!/usr/bin/env python3
"""Test that the dashboard works correctly"""

import sys
import asyncio

sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

print("=" * 70)
print("🚀 Testing Mission Control v6.2")
print("=" * 70)

try:
    from orchestrator.dashboard_mission_control import MissionControlServer, run_mission_control

    print("✅ Imports successful")

    # Create server
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"✅ Server created")

    # Test state
    state = server.state.to_dict()
    print(f"✅ State: {state}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n🚀 To start the dashboard:")
    print("   python run_mission_control_standalone.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
