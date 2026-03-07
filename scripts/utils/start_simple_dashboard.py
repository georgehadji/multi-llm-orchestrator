#!/usr/bin/env python3
"""
Simple Dashboard Starter - Με debugging
"""
import sys
import asyncio
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("🎮 Starting Mission Control LIVE v4.0")
print("=" * 60)

# Έλεγχος dependencies
print("\n📦 Έλεγχος dependencies...")
try:
    import fastapi
    print(f"   ✅ FastAPI {fastapi.__version__}")
except ImportError:
    print("   ❌ pip install fastapi")
    sys.exit(1)

try:
    import uvicorn
    print(f"   ✅ Uvicorn {uvicorn.__version__}")
except ImportError:
    print("   ❌ pip install uvicorn")
    sys.exit(1)

# Import και εκκίνηση
print("\n🚀 Εκκίνηση server...")
try:
    from orchestrator.dashboard_live import LiveDashboardServer
    
    server = LiveDashboardServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server instance created")
    print(f"   📍 URL: http://{server.host}:{server.port}")
    
    # Εκκίνηση
    print("\n🌐 Ανοίξτε το browser στο παραπάνω URL")
    print("⚠️  Πατήστε Ctrl+C για να σταματήσετε\n")
    
    asyncio.run(server.run())
    
except KeyboardInterrupt:
    print("\n\n👋 Τερματισμός...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
