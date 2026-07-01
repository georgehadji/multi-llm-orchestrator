"""
Tests for orchestrator/autonomy_config.py — Multi-Mode Selector.
"""

from __future__ import annotations

import pytest

from orchestrator.operations.autonomy_config import (
    AutonomyConfig,
    AutonomyLevel,
)


class TestAutonomyLevels:
    """Tests for AutonomyLevel enum and AutonomyConfig factory."""

    @pytest.mark.parametrize(
        "level,expected_iter,expected_repair,expected_verify",
        [
            (AutonomyLevel.LITE, 0, 0, "none"),
            (AutonomyLevel.STANDARD, 3, 1, "unit_tests"),
            (AutonomyLevel.AUTO, 5, 2, "unit_tests"),
            (AutonomyLevel.MAX, 10, 5, "behavioral"),
        ],
    )
    def test_level_presets(self, level, expected_iter, expected_repair, expected_verify):
        """Each autonomy level must have correct preset values."""
        cfg = AutonomyConfig.for_level(level)
        assert cfg.max_iterations == expected_iter
        assert cfg.repair_attempts == expected_repair
        assert cfg.verification_mode == expected_verify

    # ── Properties ──
    def test_is_lite(self):
        cfg = AutonomyConfig.for_level(AutonomyLevel.LITE)
        assert cfg.is_lite
        assert not cfg.is_standard
        assert not cfg.is_autonomous
        assert not cfg.is_max

    def test_is_autonomous_covers_auto_and_max(self):
        assert AutonomyConfig.for_level(AutonomyLevel.AUTO).is_autonomous
        assert AutonomyConfig.for_level(AutonomyLevel.MAX).is_autonomous
        assert not AutonomyConfig.for_level(AutonomyLevel.STANDARD).is_autonomous

    # ── Agent Profile Mapping ──
    @pytest.mark.parametrize(
        "profile,expected_level",
        [
            ("standard", AutonomyLevel.STANDARD),
            ("max", AutonomyLevel.MAX),
            ("creative", AutonomyLevel.AUTO),
            ("conservative", AutonomyLevel.STANDARD),
            ("research", AutonomyLevel.AUTO),
            ("unknown", AutonomyLevel.STANDARD),
        ],
    )
    def test_from_agent_profile(self, profile, expected_level):
        """Agent profile names must map to correct autonomy levels."""
        cfg = AutonomyConfig.from_agent_profile(profile)
        assert cfg.level == expected_level

    # ── Task Application ──
    def test_apply_to_task(self):
        """apply_to_task must set max_iterations and acceptance_threshold."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.AUTO)
        task = type("T", (), {"max_iterations": 0, "acceptance_threshold": 0.0})()
        cfg.apply_to_task(task)
        assert task.max_iterations == 5
        assert task.acceptance_threshold == 0.85

    # ── Model Tier Selection ──
    @pytest.mark.parametrize(
        "purpose,expected",
        [
            ("generation", "balanced"),
            ("critique", "balanced"),
            ("decomposition", "balanced"),
        ],
    )
    def test_model_tier_standard(self, purpose, expected):
        """STANDARD level must use balanced tiers."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.STANDARD)
        assert cfg.model_tier_for(purpose) == expected

    def test_model_tier_max_uses_reasoning(self):
        """MAX level must use reasoning/premium tiers."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.MAX)
        assert cfg.model_tier_for("generation") == "reasoning"
        assert cfg.model_tier_for("decomposition") == "premium"

    # ── Edge Cases ──
    def test_unknown_purpose_defaults(self):
        """Unknown purpose must return 'balanced'."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.MAX)
        assert cfg.model_tier_for("unknown") == "balanced"

    def test_lite_no_critique(self):
        """LITE mode must have zero critique passes."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.LITE)
        assert cfg.critique_passes == 0
        assert cfg.checkpoint_frequency == 0
