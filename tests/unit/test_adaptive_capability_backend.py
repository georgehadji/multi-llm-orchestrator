"""Unit tests for AdaptiveCapabilityBackend (ACR Phase 0 stub)."""

import logging
from unittest.mock import Mock

import pytest

from orchestrator.models import Model, TaskType
from orchestrator.operations.optimization import (
    AdaptiveCapabilityBackend,
    GreedyBackend,
    OptimizationBackend,
)
from orchestrator.policy import ModelProfile


def _profiles_fixture() -> dict[Model, ModelProfile]:
    models = list(Model)[:3]
    profiles = {}
    for i, m in enumerate(models):
        profiles[m] = ModelProfile(
            model=m,
            provider="test",
            cost_per_1m_input=1.0,
            cost_per_1m_output=1.0,
            quality_score=0.5 + i * 0.1,
            trust_factor=1.0,
            capable_task_types={TaskType.CODE_GEN: 0},
        )
    return profiles


def _cost_fn(profile: ModelProfile, task_type: TaskType) -> float:
    return 1.0


@pytest.mark.unit
def test_select_delegates_to_greedy_identical_pick():
    # Arrange
    profiles = _profiles_fixture()
    candidates = list(profiles.keys())
    greedy = GreedyBackend()
    acr = AdaptiveCapabilityBackend()

    # Act
    greedy_pick = greedy.select(candidates, profiles, TaskType.CODE_GEN, _cost_fn)
    acr_pick = acr.select(candidates, profiles, TaskType.CODE_GEN, _cost_fn)

    # Assert
    assert acr_pick == greedy_pick


@pytest.mark.unit
def test_select_empty_candidates_returns_none():
    # Arrange
    acr = AdaptiveCapabilityBackend()
    profiles = _profiles_fixture()

    # Act
    result = acr.select([], profiles, TaskType.CODE_GEN, _cost_fn)

    # Assert
    assert result is None


@pytest.mark.unit
def test_select_logs_decision(caplog):
    # Arrange
    profiles = _profiles_fixture()
    candidates = list(profiles.keys())
    acr = AdaptiveCapabilityBackend(mode="shadow")

    # Act
    with caplog.at_level(logging.INFO, logger="orchestrator.optimization"):
        acr.select(candidates, profiles, TaskType.CODE_GEN, _cost_fn)

    # Assert
    assert any("acr_decision" in record.message for record in caplog.records)
    assert any("mode=shadow" in record.message for record in caplog.records)


@pytest.mark.unit
def test_select_uses_injected_delegate_not_new_greedy_instance():
    # Arrange
    profiles = _profiles_fixture()
    candidates = list(profiles.keys())
    mock_delegate = Mock(spec=OptimizationBackend)
    mock_delegate.select.return_value = candidates[0]
    acr = AdaptiveCapabilityBackend(delegate=mock_delegate)

    # Act
    result = acr.select(candidates, profiles, TaskType.CODE_GEN, _cost_fn)

    # Assert
    mock_delegate.select.assert_called_once_with(candidates, profiles, TaskType.CODE_GEN, _cost_fn)
    assert result == candidates[0]
