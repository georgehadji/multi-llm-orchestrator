"""Simple starter - bypasses all import issues"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load env first
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✅ .env loaded")
except:
    print("⚠️ .env not loaded")

print("Starting Mission Control...")
print("=" * 50)

try:
    from orchestrator.dashboard_mission_control import run_mission_control
    run_mission_control(host="127.0.0.1", port=8888, open_browser=False)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
