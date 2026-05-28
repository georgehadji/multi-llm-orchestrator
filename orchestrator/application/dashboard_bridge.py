"""
Null-safe bridge to the optional dashboard integration.

P3-5 of REFACTORING_PLAN_V7.md — extracted from engine._notify_dashboard_*.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DashboardBridge:
    """Wraps the optional dashboard integration with null-safety and error suppression.

    All methods are no-ops when the underlying ``dashboard`` is ``None``,
    and swallow any exceptions the dashboard raises (dashboard failures must
    never propagate into orchestration logic).
    """

    def __init__(self, dashboard: Any) -> None:
        self._dashboard = dashboard

    # ------------------------------------------------------------------ #
    # Notifications
    # ------------------------------------------------------------------ #

    def on_project_start(self, project_id: str, state: Any, architecture_rules: Any = None) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_project_start(project_id, state, architecture_rules)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)

    def on_task_start(self, task_id: str, task: Any, model: Any) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_task_start(task_id, task, model)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)

    def on_task_progress(self, iteration: int, score: float) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_task_progress(iteration, score)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)

    def on_task_complete(self, task_id: str, status: str) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_task_complete(task_id, status)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)

    def on_model_success(self, model: Any) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_model_success(model)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)

    def on_model_failure(self, model: Any) -> None:
        if self._dashboard is None:
            return
        try:
            self._dashboard.on_model_failure(model)
        except Exception as exc:
            logger.debug("Dashboard notification failed: %s", exc)
