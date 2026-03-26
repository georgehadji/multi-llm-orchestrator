#!/usr/bin/env python3
"""Quick test of server startup"""
import sys
import os

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

print("Testing imports...")
try:
    print("1. Importing log_config...")
    from log_config import get_logger
    print("   ✓ log_config imported")
    
    print("2. Importing dashboard...")
    from dashboard_mission_control import MissionControlServer
    print("   ✓ dashboard_mission_control imported")
    
    print("\n3. Creating server...")
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✓ Server created: {server.host}:{server.port}")
    
    print("\n" + "="*50)
    print("SUCCESS! All imports working.")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
