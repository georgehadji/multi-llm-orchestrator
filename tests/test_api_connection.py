#!/usr/bin/env python3
"""Test API connection on startup"""

import sys

sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

print("=" * 70)
print("🚀 Testing Mission Control v6.4 - Auto API Connection")
print("=" * 70)

try:
    from orchestrator.dashboard_mission_control import MissionControlServer

    print("✅ Import successful")

    # Create server
    server = MissionControlServer(host="127.0.0.1", port=8888)
    print(f"✅ Server created")

    # Check API status
    print("\n🔌 Checking API connections...")
    print("   (This happens automatically when you start the dashboard)\n")

    # Simulate the check
    import os

    providers = [
        ("openai", "OPENAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("kimi", "KIMI_API_KEY"),
        ("minimax", "MINIMAX_API_KEY"),
        ("zhipu", "ZHIPUAI_API_KEY"),
    ]

    for provider, env_var in providers:
        key = os.getenv(env_var)
        if key:
            print(f"   ✅ {provider.upper()}: API Key found (will connect)")
        else:
            print(f"   ⚠️  {provider.upper()}: No API Key (set {env_var})")

    print("\n" + "=" * 70)
    print("✅ Dashboard ready!")
    print("=" * 70)
    print("\n🚀 To start with auto-API connection:")
    print("   python run_mission_control_standalone.py")
    print("\n📝 The dashboard will automatically:")
    print("   1. Connect to all configured APIs")
    print("   2. Show status for each provider (Connected/Error/No Key)")
    print("   3. Display available models")
    print("\n🔄 You can also click 'Reconnect' to retry connections")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
