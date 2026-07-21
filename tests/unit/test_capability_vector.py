"""Unit tests for CapabilityVector and its ModelProfile integration."""

import dataclasses

import pytest

pytestmark = pytest.mark.unit

from orchestrator.domain.capability_vector import CapabilityVector
from orchestrator.models import Model
from orchestrator.policy import ModelProfile


def _make_vector(**overrides) -> CapabilityVector:
    defaults = {
        "max_context": 128_000,
        "vision": False,
        "tool_use": True,
        "json_mode": True,
        "price": 1.0,
    }
    defaults.update(overrides)
    return CapabilityVector(**defaults)


@pytest.mark.unit
def test_capability_vector_is_frozen():
    # Arrange
    vector = _make_vector()

    # Act / Assert
    with pytest.raises(dataclasses.FrozenInstanceError):
        vector.max_context = 256_000  # type: ignore[misc]


@pytest.mark.unit
def test_capability_vector_derived_dims_default_none():
    # Arrange / Act
    vector = _make_vector()

    # Assert
    assert vector.reasoning is None
    assert vector.legal is None
    assert vector.coding is None
    assert vector.citation is None


@pytest.mark.unit
def test_model_profile_capability_defaults_none():
    # Arrange / Act
    profile = ModelProfile(
        model=next(iter(Model)),
        provider="test",
        cost_per_1m_input=1.0,
        cost_per_1m_output=1.0,
    )

    # Assert
    assert profile.capability is None


@pytest.mark.unit
def test_model_profile_accepts_capability_vector():
    # Arrange
    vector = _make_vector()

    # Act
    profile = ModelProfile(
        model=next(iter(Model)),
        provider="test",
        cost_per_1m_input=1.0,
        cost_per_1m_output=1.0,
        capability=vector,
    )

    # Assert
    assert profile.capability is vector
