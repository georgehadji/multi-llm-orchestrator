"""
Tests for phase_policy — the single source of truth for per-phase reasoning,
thinking, and temperature settings.

RED first.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.domain.phase_policy import (
    Phase,
    PhasePolicy,
    ReasoningEffort,
    policy_for,
    temperature_for,
    use_thinking_for,
)
from orchestrator.models import TaskType

# ── Policy completeness ───────────────────────────────────────────────────────


class TestPolicyCompleteness:
    @pytest.mark.parametrize("phase", list(Phase))
    def test_every_phase_has_a_policy(self, phase):
        p = policy_for(phase)
        assert isinstance(p, PhasePolicy)

    @pytest.mark.parametrize("phase", list(Phase))
    def test_temperature_in_valid_range(self, phase):
        assert 0.0 <= policy_for(phase).temperature <= 1.0

    @pytest.mark.parametrize("phase", list(Phase))
    def test_effort_is_enum(self, phase):
        assert isinstance(policy_for(phase).reasoning_effort, ReasoningEffort)


# ── Reasoning-heavy phases prefer reasoning + thinking ────────────────────────


class TestReasoningPhases:
    @pytest.mark.parametrize("phase", [Phase.DECOMPOSE, Phase.CRITIQUE, Phase.EVALUATE])
    def test_verification_phases_use_thinking_high(self, phase):
        p = policy_for(phase)
        assert p.use_thinking is True
        assert p.prefer_reasoning is True
        assert p.reasoning_effort is ReasoningEffort.HIGH

    @pytest.mark.parametrize("phase", [Phase.EXTRACT, Phase.SUMMARIZE, Phase.CREATIVE])
    def test_simple_phases_skip_thinking(self, phase):
        p = policy_for(phase)
        assert p.use_thinking is False
        assert p.prefer_reasoning is False
        assert p.reasoning_effort is ReasoningEffort.NONE


# ── Temperature discipline ────────────────────────────────────────────────────


class TestTemperatureValues:
    def test_extract_is_deterministic(self):
        assert policy_for(Phase.EXTRACT).temperature == 0.0

    def test_evaluate_is_low(self):
        assert policy_for(Phase.EVALUATE).temperature <= 0.1

    def test_critique_is_low(self):
        assert policy_for(Phase.CRITIQUE).temperature <= 0.1

    def test_creative_is_high(self):
        assert policy_for(Phase.CREATIVE).temperature >= 0.7

    def test_generate_is_low_for_code(self):
        assert policy_for(Phase.GENERATE).temperature <= 0.3


# ── Helper functions ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_temperature_for_returns_phase_default(self):
        assert temperature_for(Phase.EVALUATE) == policy_for(Phase.EVALUATE).temperature

    def test_task_type_override_creative(self):
        # Generate phase + creative task → high temperature, not code's low temp.
        t = temperature_for(Phase.GENERATE, task_type=TaskType.WRITING)
        assert t >= 0.7

    def test_task_type_override_extract(self):
        t = temperature_for(Phase.GENERATE, task_type=TaskType.DATA_EXTRACT)
        assert t == 0.0

    def test_use_thinking_for_evaluate(self):
        assert use_thinking_for(Phase.EVALUATE) is True

    def test_use_thinking_for_summarize(self):
        assert use_thinking_for(Phase.SUMMARIZE) is False


# ── Immutability ──────────────────────────────────────────────────────────────


class TestPolicyImmutable:
    def test_policy_is_frozen(self):
        p = policy_for(Phase.GENERATE)
        with pytest.raises((AttributeError, TypeError)):
            p.temperature = 0.99  # type: ignore[misc]


# ── Reasoning roster freshness ────────────────────────────────────────────────


class TestReasoningRoster:
    """REASONING_MODELS must track the 2026 catalogue: no dead ids, current models."""

    def test_dead_grok_4_mini_removed(self):
        from orchestrator.domain.model_registry import ModelRegistry

        assert not ModelRegistry.is_reasoning_model(
            "x-ai/grok-4-mini"
        ), "grok-4-mini is dead (not in enum, not live) — must not be a reasoning model"

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai/gpt-5.2",
            "x-ai/grok-4.3",
            "minimax/minimax-m3",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v3.2",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "qwen/qwen3-max-thinking",
        ],
    )
    def test_2026_reasoning_models_classified(self, model_id):
        from orchestrator.domain.model_registry import ModelRegistry

        assert ModelRegistry.is_reasoning_model(
            model_id
        ), f"{model_id} is a 2026 reasoning model and must be classified as such"

    @pytest.mark.parametrize(
        "model_id",
        ["openai/gpt-4o-mini", "google/gemini-3.5-flash", "z-ai/glm-4.7-flash"],
    )
    def test_simple_models_not_reasoning(self, model_id):
        from orchestrator.domain.model_registry import ModelRegistry

        assert not ModelRegistry.is_reasoning_model(model_id)
