# DEPRECATED: Replaced by unified_dashboard.py
# This file is kept for backward compatibility
# Please use: from orchestrator import run_unified_dashboard

from .unified_dashboard import (
    UnifiedDashboardServer as DashboardServer,
    run_unified_dashboard as run_dashboard,
)

__all__ = ["DashboardServer", "run_dashboard"]
