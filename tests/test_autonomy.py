"""Tests for AutonomyConfig."""
from __future__ import annotations
import pytest


class TestAutonomyLevel:
    def test_three_levels_exist(self):
        from orchestrator.autonomy import AutonomyLevel
        assert AutonomyLevel.MANUAL is not None
        assert AutonomyLevel.SUPERVISED is not None
        assert AutonomyLevel.FULL is not None

    def test_levels_ordered_by_autonomy(self):
        from orchestrator.autonomy import AutonomyLevel
        levels = list(AutonomyLevel)
        assert len(levels) == 3


class TestAutonomyConfig:
    def test_dataclass_creation(self):
        from orchestrator.autonomy import AutonomyConfig, AutonomyLevel
        cfg = AutonomyConfig(
            level=AutonomyLevel.SUPERVISED,
            require_approval_above_usd=1.0,
            require_approval_for_external_calls=False,
            max_retries_without_approval=3,
        )
        assert cfg.level == AutonomyLevel.SUPERVISED
        assert cfg.require_approval_above_usd == 1.0

    def test_dataclass_immutable_fields(self):
        from orchestrator.autonomy import AutonomyConfig, AutonomyLevel
        cfg = AutonomyConfig(
            level=AutonomyLevel.FULL,
            require_approval_above_usd=999.0,
            require_approval_for_external_calls=False,
            max_retries_without_approval=10,
        )
        assert cfg.max_retries_without_approval == 10


class TestAutonomyPresets:
    def test_all_presets_exist(self):
        from orchestrator.autonomy import AUTONOMY_PRESETS
        assert "manual" in AUTONOMY_PRESETS
        assert "supervised" in AUTONOMY_PRESETS
        assert "full" in AUTONOMY_PRESETS

    def test_manual_preset_strictest(self):
        from orchestrator.autonomy import AUTONOMY_PRESETS
        manual = AUTONOMY_PRESETS["manual"]
        assert manual.require_approval_above_usd == 0.0
        assert manual.require_approval_for_external_calls is True
        assert manual.max_retries_without_approval == 0

    def test_full_preset_most_permissive(self):
        from orchestrator.autonomy import AUTONOMY_PRESETS
        full = AUTONOMY_PRESETS["full"]
        assert full.require_approval_for_external_calls is False
        assert full.max_retries_without_approval == 10


class TestGetAutonomyConfig:
    def test_returns_preset(self):
        from orchestrator.autonomy import get_autonomy_config, AutonomyLevel
        cfg = get_autonomy_config("supervised")
        assert cfg.level == AutonomyLevel.SUPERVISED

    def test_raises_for_unknown_preset(self):
        from orchestrator.autonomy import get_autonomy_config
        with pytest.raises(ValueError, match="Unknown autonomy preset"):
            get_autonomy_config("nonexistent")


class TestRequiresApproval:
    def test_manual_always_requires_approval(self):
        from orchestrator.autonomy import requires_approval, AUTONOMY_PRESETS
        cfg = AUTONOMY_PRESETS["manual"]
        assert requires_approval(cfg, estimated_cost_usd=0.001) is True

    def test_full_never_requires_approval_for_low_cost(self):
        from orchestrator.autonomy import requires_approval, AUTONOMY_PRESETS
        cfg = AUTONOMY_PRESETS["full"]
        assert requires_approval(cfg, estimated_cost_usd=0.50) is False

    def test_supervised_requires_approval_above_threshold(self):
        from orchestrator.autonomy import requires_approval, AUTONOMY_PRESETS
        cfg = AUTONOMY_PRESETS["supervised"]
        assert requires_approval(cfg, estimated_cost_usd=5.00) is True
        assert requires_approval(cfg, estimated_cost_usd=0.50) is False
