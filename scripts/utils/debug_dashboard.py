#!/usr/bin/env python3
"""
Debug Dashboard - Εύρεση προβλημάτων
"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("🔍 Dashboard Debug")
print("=" * 60)

# Έλεγχος 1: Imports
print("\n1️⃣ Έλεγχος imports...")
try:
    from orchestrator import LiveDashboardServer, run_live_dashboard
    print("   ✅ LiveDashboardServer imported")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Έλεγχος 2: Dependencies
print("\n2️⃣ Έλεγχος dependencies...")
try:
    import fastapi
    print(f"   ✅ FastAPI: {fastapi.__version__}")
except ImportError:
    print("   ❌ FastAPI not installed")
    print("   💡 Run: pip install fastapi uvicorn websockets")

try:
    import uvicorn
    print(f"   ✅ Uvicorn: {uvicorn.__version__}")
except ImportError:
    print("   ❌ Uvicorn not installed")

try:
    import websockets
    print(f"   ✅ Websockets: {websockets.__version__}")
except ImportError:
    print("   ⚠️  Websockets not installed (optional για WebSocket)")

# Έλεγχος 3: Δημιουργία server instance
print("\n3️⃣ Δημιουργία server instance...")
try:
    server = LiveDashboardServer(host="127.0.0.1", port=8888)
    print(f"   ✅ Server created: {server}")
    print(f"   📍 Host: {server.host}, Port: {server.port}")
except Exception as e:
    print(f"   ❌ Error creating server: {e}")
    import traceback
    traceback.print_exc()

# Έλεγχος 4: State
print("\n4️⃣ Έλεγχος state...")
try:
    from orchestrator.dashboard_live import DashboardState
    state = DashboardState()
    print(f"   ✅ DashboardState created")
    print(f"   📊 Initial state: {state.status}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ Debug complete!")
print("=" * 60)
print("\n🚀 Για να ξεκινήσεις το dashboard:")
print("   python -c \"from orchestrator import run_live_dashboard; run_live_dashboard()\"")
print("\n🌐 URL: http://127.0.0.1:8888")
print("⚠️  Άφησε το terminal ανοιχτό όσο τρέχει το dashboard!")
