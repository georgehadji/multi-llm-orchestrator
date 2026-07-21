"""Unit tests for ConstraintPlanner backend/shadow-backend wiring (ACR Phase 0)."""

from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import Model, TaskType
from orchestrator.planner import ConstraintPlanner
from orchestrator.policy import ModelProfile
from orchestrator.policy_engine import PolicyEngine
from orchestrator.optimization import OptimizationBackend


def _profiles_fixture() -> dict[Model, ModelProfile]:
    models = list(Model)[:3]
    return {
        m: ModelProfile(
            model=m,
            provider="test",
            cost_per_1m_input=1.0,
            cost_per_1m_output=1.0,
            capable_task_types={TaskType.CODE_GEN: 0},
        )
        for m in models
    }


def _make_planner(**kwargs) -> ConstraintPlanner:
    profiles = _profiles_fixture()
    api_health = dict.fromkeys(profiles, True)
    return ConstraintPlanner(
        profiles=profiles,
        policy_engine=PolicyEngine(),
        api_health=api_health,
        **kwargs,
    )


@pytest.mark.unit
def test_set_backend_swaps_active_backend():
    # Arrange
    planner = _make_planner()
    mock_backend = Mock(spec=OptimizationBackend)
    mock_backend.select.return_value = list(planner._profiles.keys())[0]

    # Act
    planner.set_backend(mock_backend)
    planner.select_model(TaskType.CODE_GEN, policies=[], budget_remaining=100.0)

    # Assert
    mock_backend.select.assert_called_once()


@pytest.mark.unit
def test_shadow_backend_does_not_affect_return_value():
    # Arrange
    planner_plain = _make_planner()
    planner_shadowed = _make_planner()
    mock_shadow = Mock(spec=OptimizationBackend)
    mock_shadow.select.return_value = None
    planner_shadowed.set_shadow_backend(mock_shadow)

    # Act
    plain_result = planner_plain.select_model(
        TaskType.CODE_GEN, policies=[], budget_remaining=100.0
    )
    shadowed_result = planner_shadowed.select_model(
        TaskType.CODE_GEN, policies=[], budget_remaining=100.0
    )

    # Assert
    assert shadowed_result == plain_result
    mock_shadow.select.assert_called_once()


@pytest.mark.unit
def test_shadow_backend_exception_does_not_propagate():
    # Arrange
    planner = _make_planner()
    mock_shadow = Mock(spec=OptimizationBackend)
    mock_shadow.select.side_effect = RuntimeError("boom")
    planner.set_shadow_backend(mock_shadow)

    # Act
    result = planner.select_model(TaskType.CODE_GEN, policies=[], budget_remaining=100.0)

    # Assert — no exception escaped, a real result still returned
    assert result is not None


@pytest.mark.unit
def test_no_shadow_backend_is_noop():
    # Arrange
    planner = _make_planner()

    # Act / Assert — no error, normal selection proceeds
    result = planner.select_model(TaskType.CODE_GEN, policies=[], budget_remaining=100.0)
    assert result is not None
