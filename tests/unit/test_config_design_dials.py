"""Unit tests for taste-skill feature flags and design dials in config."""

import pytest


@pytest.mark.unit
def test_taste_skill_enabled_default_true(monkeypatch):
    monkeypatch.delenv("ORCH_TASTE_SKILL_ENABLED", raising=False)
    # Re-import to pick up cleared env
    import importlib
    import orchestrator.crosscutting.config as cfg_mod

    fresh = cfg_mod.FeatureFlags()
    assert fresh.taste_skill_enabled is True


@pytest.mark.unit
def test_image_reference_pipeline_default_false(monkeypatch):
    monkeypatch.delenv("ORCH_IMAGE_REFERENCE_PIPELINE", raising=False)
    import orchestrator.crosscutting.config as cfg_mod

    fresh = cfg_mod.FeatureFlags()
    assert fresh.image_reference_pipeline is False


@pytest.mark.unit
def test_dials_default_to_5(monkeypatch):
    for var in ("ORCH_DESIGN_VARIANCE", "ORCH_MOTION_INTENSITY", "ORCH_VISUAL_DENSITY"):
        monkeypatch.delenv(var, raising=False)
    import orchestrator.crosscutting.config as cfg_mod

    fresh = cfg_mod.OrchestratorSettings()
    assert fresh.design_variance == 5
    assert fresh.motion_intensity == 5
    assert fresh.visual_density == 5


@pytest.mark.unit
def test_dials_read_from_env(monkeypatch):
    monkeypatch.setenv("ORCH_DESIGN_VARIANCE", "8")
    monkeypatch.setenv("ORCH_MOTION_INTENSITY", "3")
    monkeypatch.setenv("ORCH_VISUAL_DENSITY", "9")
    import orchestrator.crosscutting.config as cfg_mod

    fresh = cfg_mod.OrchestratorSettings()
    assert fresh.design_variance == 8
    assert fresh.motion_intensity == 3
    assert fresh.visual_density == 9


@pytest.mark.unit
def test_dial_below_min_raises(monkeypatch):
    monkeypatch.setenv("ORCH_DESIGN_VARIANCE", "0")
    import orchestrator.crosscutting.config as cfg_mod
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cfg_mod.OrchestratorSettings()


@pytest.mark.unit
def test_dial_above_max_raises(monkeypatch):
    monkeypatch.setenv("ORCH_MOTION_INTENSITY", "11")
    import orchestrator.crosscutting.config as cfg_mod
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cfg_mod.OrchestratorSettings()
