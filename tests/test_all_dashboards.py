#!/usr/bin/env python3
"""
Test all dashboard imports
"""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 60)
print("Testing All Dashboard Imports")
print("=" * 60)

dashboards = [
    ("dashboard_live", "run_live_dashboard", "LIVE v4.0"),
    ("dashboard_antd", "run_ant_design_dashboard", "Ant Design v3.0"),
    ("dashboard_enhanced", "run_enhanced_dashboard", "Enhanced v2.0"),
    ("dashboard_real", "run_dashboard_realtime", "Real-time v5.1"),
]

all_ok = True

for module_name, func_name, display_name in dashboards:
    try:
        module = __import__(f"orchestrator.{module_name}", fromlist=[func_name])
        func = getattr(module, func_name)
        print(f"[OK] {display_name:20} - {func_name}() ready")
    except Exception as e:
        print(f"[ERROR] {display_name:20} - Error: {e}")
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("All dashboards ready!")
    print("\nStart with:")
    print("   python start_dashboard.py live")
    print("   python start_dashboard.py antd")
else:
    print("Some dashboards have errors")
print("=" * 60)
