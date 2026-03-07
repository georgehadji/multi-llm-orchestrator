#!/usr/bin/env python3
"""Verify Mission Control imports work"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("="*60)
print("🔍 Testing Mission Control Imports")
print("="*60)

try:
    print("\n1. Importing MissionControlServer...")
    from orchestrator.dashboard_mission_control import MissionControlServer
    print(f"   ✅ Success: {MissionControlServer}")
    
    print("\n2. Creating instance...")
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Instance created")
    
    print("\n3. Checking state...")
    state = server.state.to_dict()
    print(f"   ✅ Version: {state['version']}")
    print(f"   ✅ Status: {state['status']}")
    
    print("\n4. Checking HTML...")
    html = server._get_html()
    print(f"   ✅ HTML length: {len(html):,} chars")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🚀 Ready to run:")
    print("   python start_mission_control.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("❌ TESTS FAILED")
    print("="*60)
