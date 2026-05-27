"""
DashboardBridge — Publishes orchestrator events to the dashboard.
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the ENGINE_OPTIMIZATION_PLAN Phase 3: extracts 6 dashboard-related
methods from Orchestrator into a self-contained bridge class.

Usage:
    bridge = DashboardBridge(orch)
    bridge.set_dashboard_integration(integration)
    bridge.notify_project_start("my-project", {"tasks": 5})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Model, Task, TaskResult

logger = logging.getLogger(__name__)


class DashboardBridge:
    """Publishes orchestrator state changes to the dashboard integration."""

    def __init__(self, orchestrator: object):
        self._orch = orchestrator

    @property
    def _dashboard(self) -> object | None:
        return getattr(self._orch, "_dashboard_integration", None)

    def set_dashboard_integration(self, integration: object) -> None:
        """Set dashboard integration for real-time updates."""
        self._orch._dashboard_integration = integration

    def notify_project_start(self, project_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Notify dashboard of project start."""
        dash = self._dashboard
        if dash and hasattr(dash, "on_project_start"):
            try:
                dash.on_project_start(project_id, metadata or {})
            except Exception:
                logger.debug("Dashboard project start notification failed", exc_info=True)

    def notify_task_start(self, task_id: str, task_name: str, project_id: str) -> None:
        """Notify dashboard of task start."""
        dash = self._dashboard
        if dash and hasattr(dash, "on_task_start"):
            try:
                dash.on_task_start(task_id, task_name, project_id)
            except Exception:
                logger.debug("Dashboard task start notification failed", exc_info=True)

    def notify_task_progress(
        self, task_id: str, progress: float, status: str, details: str = ""
    ) -> None:
        """Notify dashboard of task progress."""
        dash = self._dashboard
        if dash and hasattr(dash, "on_task_progress"):
            try:
                dash.on_task_progress(task_id, progress, status, details)
            except Exception:
                logger.debug("Dashboard task progress notification failed", exc_info=True)

    def notify_task_complete(self, task_id: str, result: TaskResult, quality_score: float) -> None:
        """Notify dashboard of task completion."""
        dash = self._dashboard
        if dash and hasattr(dash, "on_task_complete"):
            try:
                dash.on_task_complete(task_id, result, quality_score)
            except Exception:
                logger.debug("Dashboard task complete notification failed", exc_info=True)

    def build_metrics_dict(self) -> dict[str, dict[str, Any]]:
        """Build a per-model metrics dict from live ModelProfile data."""
        profiles = getattr(self._orch, "_profiles", {})
        return {
            model.value: {
                "avg_latency_ms": getattr(p, "avg_latency_ms", 0),
                "quality_score": getattr(p, "quality_score", 0.0),
                "trust_factor": getattr(p, "trust_factor", 1.0),
                "call_count": getattr(p, "call_count", 0),
                "error_rate": getattr(p, "error_rate", lambda: 0.0)(),
                "success_rate": getattr(p, "success_rate", 0.0),
            }
            for model, p in profiles.items()
        }
