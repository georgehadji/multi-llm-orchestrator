#!/usr/bin/env python3
"""Test Enhanced Dashboard v2.0."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 70)
print("Enhanced Dashboard v2.0 - Import Test")
print("=" * 70)

try:
    from orchestrator import (
        EnhancedDashboardServer,
        EnhancedDataProvider,
        DashboardIntegration,
        ArchitectureInfo,
        ProjectInfo,
        ActiveTaskInfo,
        ModelStatusInfo,
        run_enhanced_dashboard,
    )
    print("[OK] All Enhanced Dashboard imports successful")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
