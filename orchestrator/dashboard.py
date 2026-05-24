# DEPRECATED: Dashboard variants consolidated into dashboard_core/mission_control.py
# This file is kept for backward compatibility.
# Please use: from orchestrator.dashboard_core.mission_control import MissionControlView

from .dashboard_core.mission_control import (
    MissionControlView as DashboardServer,
    create_view as run_dashboard,
)

__all__ = ["DashboardServer", "run_dashboard"]
