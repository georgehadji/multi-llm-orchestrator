"""
Start dashboard with debug output
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🚀 STARTING MISSION CONTROL (DEBUG MODE)")
print("=" * 60)

try:
    print("\n1. Importing dashboard...")
    from orchestrator.dashboard_mission_control import MissionControlServer, run_mission_control
    print("   ✅ Import successful")
    
    print("\n2. Creating server...")
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server created (version {server.state.version})")
    
    print("\n3. Starting server...")
    print("   Press Ctrl+C to stop\n")
    
    import asyncio
    asyncio.run(server.run())
    
except KeyboardInterrupt:
    print("\n\n👋 Server stopped by user")
except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    print("\n" + "=" * 60)
    print("FULL TRACEBACK:")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)
    input("\nPress Enter to exit...")
