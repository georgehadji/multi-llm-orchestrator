"""Unit tests for container._wire_acr_backend (ACR Phase 0 flag wiring)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orchestrator.engine_core.container import _wire_acr_backend
from orchestrator.operations.optimization import AdaptiveCapabilityBackend


@pytest.mark.unit
def test_flag_off_does_not_call_set_backend():
    # Arrange
    planner = Mock()
    flags = SimpleNamespace(acr_backend="off")

    # Act
    _wire_acr_backend(planner, flags)

    # Assert
    planner.set_backend.assert_not_called()
    planner.set_shadow_backend.assert_not_called()


@pytest.mark.unit
def test_flag_on_calls_set_backend_with_adaptive_backend():
    # Arrange
    planner = Mock()
    flags = SimpleNamespace(acr_backend="on")

    # Act
    _wire_acr_backend(planner, flags)

    # Assert
    planner.set_backend.assert_called_once()
    (backend_arg,), _ = planner.set_backend.call_args
    assert isinstance(backend_arg, AdaptiveCapabilityBackend)
    planner.set_shadow_backend.assert_not_called()


@pytest.mark.unit
def test_flag_shadow_calls_set_shadow_backend_only():
    # Arrange
    planner = Mock()
    flags = SimpleNamespace(acr_backend="shadow")

    # Act
    _wire_acr_backend(planner, flags)

    # Assert
    planner.set_shadow_backend.assert_called_once()
    (backend_arg,), _ = planner.set_shadow_backend.call_args
    assert isinstance(backend_arg, AdaptiveCapabilityBackend)
    planner.set_backend.assert_not_called()
