# DEPRECATED: Replaced by unified_dashboard.py
# This file is kept for backward compatibility
# Please use: from orchestrator import run_unified_dashboard

from .unified_dashboard import (
    UnifiedDashboardServer as OptimizedDashboardServer,
)
from .unified_dashboard import (
    run_unified_dashboard as run_optimized_dashboard,
)

__all__ = ["OptimizedDashboardServer", "run_optimized_dashboard"]
