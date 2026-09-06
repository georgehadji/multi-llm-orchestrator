# DEPRECATED: Dashboard variants consolidated into dashboard_core/mission_control.py
# This file is kept for backward compatibility.
# Please use: from orchestrator.dashboard_core.mission_control import MissionControlView

from .dashboard_core import run_dashboard
from .dashboard_core.mission_control import MissionControlView as DashboardServer

__all__ = ["DashboardServer", "run_dashboard"]
