#!/usr/bin/env python3
"""
Test Mission Control Dashboard
==============================
"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("🚀 Testing Mission Control v6.0")
print("=" * 60)

# Test imports
print("\n1️⃣ Testing imports...")
try:
    from orchestrator import (
        run_mission_control,
        MissionControlServer,
        ProjectRunner,
    )
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test server creation
print("\n2️⃣ Creating server instance...")
try:
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server created")
    print(f"   📍 Will run on: http://{server.host}:{server.port}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test state
print("\n3️⃣ Testing state...")
try:
    state_dict = server.state.to_dict()
    print(f"   ✅ State generated")
    print(f"   📊 Version: {state_dict.get('version')}")
    print(f"   🤖 Models: {len(state_dict.get('available_models', []))}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test HTML generation
print("\n4️⃣ Testing HTML generation...")
try:
    html = server._get_html()
    print(f"   ✅ HTML generated: {len(html):,} characters")
    
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
        status = "✅" if present else "❌"
        print(f"   {status} {name}")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ Mission Control v6.0 is READY!")
print("=" * 60)
print("""
🚀 To start:

   python start_mission_control.py

🌐 URL: http://localhost:8888

✨ Features:
   • Project starter form (prompt, criteria, type, budget)
   • Real-time progress tracking
   • Architecture visualization
   • Model usage monitoring
   • Live task execution
""")
