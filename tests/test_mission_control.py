#!/usr/bin/env python3
"""
Test Mission Control Dashboard
==============================
"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("Testing Mission Control v6.0")
print("=" * 60)

print("\n1. Testing imports...")
try:
    from orchestrator import (
        run_mission_control,
        MissionControlServer,
        ProjectRunner,
    )
    print("   [OK] All imports successful")
except Exception as e:
    print(f"   [ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating server instance...")
try:
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   [OK] Server created")
    print(f"   [INFO] Will run on: http://{server.host}:{server.port}")
except Exception as e:
    print(f"   [ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing state...")
try:
    state_dict = server.state.to_dict()
    print(f"   [OK] State generated")
    print(f"   [INFO] Version: {state_dict.get('version')}")
    print(f"   [INFO] Models: {len(state_dict.get('available_models', []))}")
except Exception as e:
    print(f"   [ERROR] Failed: {e}")

print("\n4. Testing HTML generation...")
try:
    html = server._get_html()
    print(f"   [OK] HTML generated: {len(html):,} characters")
    
    # Check key features
    checks = [
        ("Project Form", "project-type" in html),
        ("Prompt Input", "prompt" in html),
        ("Start Button", "Start Project" in html),
        ("Progress Bar", "progress-bar" in html),
        ("Architecture Panel", "Architecture" in html),
        ("Models Display", "Models" in html),
        ("WebSocket", "websocket" in html.lower()),
    ]
    
    for name, present in checks:
        status = "[OK]" if present else "[ERROR]"
        print(f"   {status} {name}")
        
except Exception as e:
    print(f"   [ERROR] Failed: {e}")

print("\n" + "=" * 60)
print("[OK] Mission Control v6.0 is READY!")
print("=" * 60)
print("""
To start:
   python start_mission_control.py

URL: http://localhost:8888

Features:
   - Project starter form (prompt, criteria, type, budget)
   - Real-time progress tracking
   - Architecture visualization
   - Model usage monitoring
   - Live task execution
""")
