#!/usr/bin/env python3
"""Test Unified Dashboard - moved to tests/ folder"""

import sys

sys.path.insert(0, r"E:\Documents\Vibe-Coding\Ai Orchestrator")

print("=" * 60)
print("🚀 Testing Unified Dashboard v5.2")
print("=" * 60)

try:
    from orchestrator import (
        run_unified_dashboard,
        UnifiedDashboardServer,
        ApiConnectionManager,
    )

    print("✅ All imports successful")

    dashboard = UnifiedDashboardServer()
    print(f"✅ Dashboard created: {dashboard.host}:{dashboard.port}")

    html = dashboard._get_html()
    print(f"✅ HTML: {len(html):,} chars")

    print("\n✅ Unified Dashboard ready!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
