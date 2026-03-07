#!/usr/bin/env python3
"""
Dashboard Starter - Με καθυστέρηση για σωστό loading
"""
import sys
import asyncio
import time
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("🎮 Mission Control LIVE v4.0")
print("=" * 60)

# Dependencies check
print("\n📦 Checking dependencies...")
try:
    import fastapi
    import uvicorn
    print(f"   ✅ FastAPI {fastapi.__version__}")
    print(f"   ✅ Uvicorn {uvicorn.__version__}")
except ImportError as e:
    print(f"   ❌ Missing: {e}")
    print("   💡 Run: pip install fastapi uvicorn websockets")
    sys.exit(1)

# Import server
print("\n🔧 Loading server...")
from orchestrator.dashboard_live import LiveDashboardServer

server = LiveDashboardServer(host="127.0.0.1", port=8888)
print(f"   ✅ Ready on http://{server.host}:{server.port}")

# Open browser with delay
import webbrowser
url = f"http://{server.host}:{server.port}"

print("\n" + "=" * 60)
print("🌐 Starting server...")
print("=" * 60)
print(f"\n📍 URL: {url}")
print("⏳ Opening browser in 2 seconds...")
print("⚠️  Press Ctrl+C to stop\n")

# Άνοιγμα browser με καθυστέρηση 2 δευτερόλεπτα
def open_browser_delayed():
    time.sleep(2)
    webbrowser.open(url)
    print("🌍 Browser opened!")

import threading
threading.Thread(target=open_browser_delayed, daemon=True).start()

# Εκκίνηση server
try:
    asyncio.run(server.run())
except KeyboardInterrupt:
    print("\n\n👋 Goodbye!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
