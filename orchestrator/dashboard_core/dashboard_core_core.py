# Compatibility shim — mission_control.py imports from here.
# Canonical source is orchestrator.dashboard_core.core.
from .core import DashboardView, ViewContext

__all__ = ["DashboardView", "ViewContext"]
