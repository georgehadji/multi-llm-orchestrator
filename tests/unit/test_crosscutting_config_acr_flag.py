"""Unit tests for the ORCH_ACR_BACKEND feature flag (ACR Phase 0 seam)."""

import pytest
from pydantic import ValidationError


@pytest.mark.unit
def test_acr_backend_default_off(monkeypatch):
    # Arrange
    monkeypatch.delenv("ORCH_ACR_BACKEND", raising=False)
    import orchestrator.crosscutting.config as cfg_mod

    # Act
    fresh = cfg_mod.FeatureFlags()

    # Assert
    assert fresh.acr_backend == "off"


@pytest.mark.unit
def test_acr_backend_reads_shadow_from_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("ORCH_ACR_BACKEND", "shadow")
    import orchestrator.crosscutting.config as cfg_mod

    # Act
    fresh = cfg_mod.FeatureFlags()

    # Assert
    assert fresh.acr_backend == "shadow"


@pytest.mark.unit
def test_acr_backend_reads_on_from_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("ORCH_ACR_BACKEND", "on")
    import orchestrator.crosscutting.config as cfg_mod

    # Act
    fresh = cfg_mod.FeatureFlags()

    # Assert
    assert fresh.acr_backend == "on"


@pytest.mark.unit
def test_acr_backend_rejects_invalid_value(monkeypatch):
    # Arrange
    monkeypatch.setenv("ORCH_ACR_BACKEND", "bogus")
    import orchestrator.crosscutting.config as cfg_mod

    # Act / Assert
    with pytest.raises(ValidationError):
        cfg_mod.FeatureFlags()
