"""
Unit tests for orchestrator.application.dashboard_bridge.DashboardBridge

P3-5 of REFACTORING_PLAN_V7.md.
"""

from unittest.mock import MagicMock

import pytest

from orchestrator.application.dashboard_bridge import DashboardBridge

# ─────────────────────────────────────────────────────────────────────────────
# Null dashboard — all methods are no-ops
# ─────────────────────────────────────────────────────────────────────────────


def test_null_dashboard_on_project_start():
    bridge = DashboardBridge(None)
    bridge.on_project_start("pid", MagicMock())  # must not raise


def test_null_dashboard_on_task_start():
    bridge = DashboardBridge(None)
    bridge.on_task_start("t1", MagicMock(), MagicMock())


def test_null_dashboard_on_task_progress():
    bridge = DashboardBridge(None)
    bridge.on_task_progress(1, 0.9)


def test_null_dashboard_on_task_complete():
    bridge = DashboardBridge(None)
    bridge.on_task_complete("t1", "completed")


# ─────────────────────────────────────────────────────────────────────────────
# Live dashboard — calls are forwarded
# ─────────────────────────────────────────────────────────────────────────────


def test_on_project_start_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    state = MagicMock()
    bridge.on_project_start("pid", state, architecture_rules="rules")
    dash.on_project_start.assert_called_once_with("pid", state, "rules")


def test_on_task_start_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    task, model = MagicMock(), MagicMock()
    bridge.on_task_start("t1", task, model)
    dash.on_task_start.assert_called_once_with("t1", task, model)


def test_on_task_progress_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    bridge.on_task_progress(3, 0.75)
    dash.on_task_progress.assert_called_once_with(3, 0.75)


def test_on_task_complete_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    bridge.on_task_complete("t1", "completed")
    dash.on_task_complete.assert_called_once_with("t1", "completed")


def test_on_model_success_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    model = MagicMock()
    bridge.on_model_success(model)
    dash.on_model_success.assert_called_once_with(model)


def test_on_model_failure_forwards():
    dash = MagicMock()
    bridge = DashboardBridge(dash)
    model = MagicMock()
    bridge.on_model_failure(model)
    dash.on_model_failure.assert_called_once_with(model)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard exceptions are swallowed
# ─────────────────────────────────────────────────────────────────────────────


def test_dashboard_exception_is_suppressed():
    dash = MagicMock()
    dash.on_task_complete.side_effect = RuntimeError("dashboard down")
    bridge = DashboardBridge(dash)
    bridge.on_task_complete("t1", "completed")  # must not raise
