"""Tests for Content Preflight Gate integration."""
import pytest
from orchestrator.hooks import EventType
from orchestrator.models import TaskResult, Model
from orchestrator.preflight import PreflightResult, PreflightAction


def test_preflight_check_event_exists():
    assert hasattr(EventType, "PREFLIGHT_CHECK")
    assert EventType.PREFLIGHT_CHECK.value == "preflight_check"


def test_task_result_has_preflight_fields():
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
    )
    assert result.preflight_passed is True
    assert result.preflight_result is None


def test_task_result_accepts_preflight_result():
    pf = PreflightResult(action=PreflightAction.WARN, passed=False, warnings=["test warning"])
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
        preflight_result=pf,
        preflight_passed=False,
    )
    assert result.preflight_passed is False
    assert result.preflight_result.action == PreflightAction.WARN
