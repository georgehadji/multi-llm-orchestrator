"""
Unified Dashboard Core
======================
Single dashboard core with pluggable views.
"""
from .core import (
    DashboardCore,
    get_dashboard_core,
    DashboardView,
    ViewContext,
    ViewRegistry,
    run_dashboard,
)
from .mission_control import MissionControlView

__all__ = [
    "DashboardCore",
    "get_dashboard_core",
    "DashboardView",
    "ViewContext",
    "ViewRegistry",
    "run_dashboard",
    "MissionControlView",
]
