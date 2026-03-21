#!/usr/bin/env python3
"""Test Ant Design Dashboard."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

from orchestrator.dashboard_antd import AntDesignDashboardServer

print("=" * 70)
print("Ant Design Dashboard - Import Test")
print("=" * 70)

try:
    # Import test
    from orchestrator import AntDesignDashboardServer, run_ant_design_dashboard
    print("[OK] AntDesignDashboardServer imported successfully")

    # Check class exists
    print(f"[OK] Server class: {AntDesignDashboardServer}")
    print(f"[OK] Run function: {run_ant_design_dashboard}")

    print("\n" + "=" * 70)
    print("Ant Design Dashboard Ready!")
    print("=" * 70)
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
