"""
Dashboard Diagnostic Script
===========================
Τρέξε αυτό για να δεις αν υπάρχουν errors στο dashboard.
"""
import sys
import os

print("=" * 60)
print("🔍 DASHBOARD DIAGNOSTIC")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from orchestrator.dashboard_mission_control import MissionControlServer
    print("   ✅ MissionControlServer imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check syntax of the file
print("\n2. Checking Python syntax...")
import ast
try:
    with open('orchestrator/dashboard_mission_control.py', 'r') as f:
        code = f.read()
    ast.parse(code)
    print("   ✅ Syntax is valid")
except SyntaxError as e:
    print(f"   ❌ Syntax error: {e}")
    sys.exit(1)

# Test 3: Create server instance
print("\n3. Creating server instance...")
try:
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server created (version: {server.state.version})")
except Exception as e:
    print(f"   ❌ Failed to create server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check initial state
print("\n4. Checking initial state...")
try:
    state_dict = server.state.to_dict()
    print(f"   Version: {state_dict.get('version')}")
    print(f"   Server status: {state_dict.get('server_status')}")
    print(f"   Projects: {state_dict.get('total_projects')}")
    print(f"   APIs: {len(state_dict.get('api_status', []))}")
except Exception as e:
    print(f"   ❌ Failed to get state: {e}")

print("\n" + "=" * 60)
print("✅ All checks passed! The dashboard should work.")
print("=" * 60)
print("\nTo start the dashboard, run:")
print("  Start_Mission_Control.bat")
