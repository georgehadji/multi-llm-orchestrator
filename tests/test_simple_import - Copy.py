"""Simple test imports"""

import sys

sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

try:
    print("Testing imports...")
    from orchestrator.dashboard_live import LiveDashboardServer

    print("✅ LiveDashboardServer imported")

    server = LiveDashboardServer()
    print(f"✅ Server created: host={server.host}, port={server.port}")
    print("\n✅ All tests passed! Try running the dashboard now.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
