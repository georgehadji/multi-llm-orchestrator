#!/usr/bin/env python3
"""Test Live Dashboard v4.0."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 70)
print("Live Dashboard v4.0 - Import Test")
print("=" * 70)

try:
    from orchestrator import (
        LiveDashboardServer,
        DashboardLiveIntegration,
        run_live_dashboard,
    )
    print("[OK] All Live Dashboard imports successful")
except Exception as e:
    print(f"[ERROR] Error: {e}")
