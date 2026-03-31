"""
Test suite for Minimax and Z.ai (GLM) model integration.

Tests verify:
- Provider detection for new models
- Cost table entries
- Routing table inclusion
- Fallback chain entries
- Profile building
"""

import pytest
from orchestrator.models import (
    Model,
    TaskType,
    get_provider,
    COST_TABLE,
    ROUTING_TABLE,
    FALLBACK_CHAIN,
    build_default_profiles,
)


class TestMinimaxProviderDetection:
    """Test get_provider() correctly identifies Minimax models."""

    def test_minimax_3_returns_minimax_provider(self):
        """Minimax-3 should be detected as 'minimax' provider."""
        assert get_provider(Model.MINIMAX_TEXT_01) == "minimax"

    def test_minimax_3_model_value_correct(self):
        """Verify MINIMAX_TEXT_01 enum value is set correctly."""
        assert Model.MINIMAX_TEXT_01.value == "MiniMax-Text-01"


class TestZaiGlmProviderDetection:
    """Test get_provider() correctly identifies Z.ai/GLM models."""

    def test_zai_glm_returns_zhipu_provider(self):
        """Z.ai GLM-4 should be detected as 'zhipu' provider."""
        assert get_provider(Model.GLM_4) == "zhipu"

    def test_zai_glm_model_value_correct(self):
        """Verify GLM_4 enum value is set correctly."""
        assert Model.GLM_4.value == "glm-4"


class TestMinimaxCostTable:
    """Test Minimax-3 has cost entries."""

    def test_minimax_3_in_cost_table(self):
        """MINIMAX_TEXT_01 must have cost entries."""
        assert Model.MINIMAX_TEXT_01 in COST_TABLE

    def test_minimax_3_has_input_cost(self):
        """MINIMAX_TEXT_01 cost entry must include input cost."""
        assert "input" in COST_TABLE[Model.MINIMAX_TEXT_01]

    def test_minimax_3_has_output_cost(self):
        """MINIMAX_TEXT_01 cost entry must include output cost."""
        assert "output" in COST_TABLE[Model.MINIMAX_TEXT_01]

    def test_minimax_3_costs_are_positive(self):
        """MINIMAX_TEXT_01 costs must be positive numbers."""
        costs = COST_TABLE[Model.MINIMAX_TEXT_01]
        assert costs["input"] > 0
        assert costs["output"] > 0


class TestZaiGlmCostTable:
    """Test Z.ai GLM-4 has cost entries."""

    def test_zai_glm_in_cost_table(self):
        """GLM_4 must have cost entries."""
        assert Model.GLM_4 in COST_TABLE

    def test_zai_glm_has_input_cost(self):
        """GLM_4 cost entry must include input cost."""
        assert "input" in COST_TABLE[Model.GLM_4]

    def test_zai_glm_has_output_cost(self):
        """GLM_4 cost entry must include output cost."""
        assert "output" in COST_TABLE[Model.GLM_4]

    def test_zai_glm_costs_are_positive(self):
        """GLM_4 costs must be positive numbers."""
        costs = COST_TABLE[Model.GLM_4]
        assert costs["input"] > 0
        assert costs["output"] > 0


class TestMinimaxRoutingTable:
    """Test Minimax appears in routing table for appropriate task types."""

    def test_minimax_in_at_least_one_task_type(self):
        """Minimax should be routed for at least one task type."""
        found = False
        for task_type, model_list in ROUTING_TABLE.items():
            if Model.MINIMAX_TEXT_01 in model_list:
                found = True
                break
        assert found, "MINIMAX_TEXT_01 not found in any ROUTING_TABLE entry"

    def test_minimax_in_reasoning_task(self):
        """Minimax should be included for reasoning tasks (strength)."""
        assert Model.MINIMAX_TEXT_01 in ROUTING_TABLE[TaskType.REASONING]

    def test_minimax_in_code_gen_task(self):
        """Minimax should be included for code generation (efficient reasoning)."""
        assert Model.MINIMAX_TEXT_01 in ROUTING_TABLE[TaskType.CODE_GEN]


class TestZaiGlmRoutingTable:
    """Test Z.ai GLM appears in routing table for appropriate task types."""

    def test_zai_glm_in_at_least_one_task_type(self):
        """Z.ai GLM should be routed for at least one task type."""
        found = False
        for task_type, model_list in ROUTING_TABLE.items():
            if Model.GLM_4 in model_list:
                found = True
                break
        assert found, "GLM_4 not found in any ROUTING_TABLE entry"

    def test_zai_glm_in_writing_task(self):
        """Z.ai GLM should be included for writing tasks (general strength)."""
        assert Model.GLM_4 in ROUTING_TABLE[TaskType.WRITING]

    def test_zai_glm_in_code_gen_task(self):
        """Z.ai GLM should be included for code generation."""
        assert Model.GLM_4 in ROUTING_TABLE[TaskType.CODE_GEN]


class TestMinimaxFallbackChain:
    """Test Minimax has fallback chain entry."""

    def test_minimax_in_fallback_chain(self):
        """MINIMAX_TEXT_01 must have a fallback chain entry."""
        assert Model.MINIMAX_TEXT_01 in FALLBACK_CHAIN

    def test_minimax_fallback_is_different_provider(self):
        """Minimax fallback should be from a different provider."""
        fallback = FALLBACK_CHAIN[Model.MINIMAX_TEXT_01]
        minimax_provider = get_provider(Model.MINIMAX_TEXT_01)
        fallback_provider = get_provider(fallback)
        assert (
            minimax_provider != fallback_provider
        ), f"Minimax fallback should be cross-provider, got {fallback_provider}"

    def test_minimax_fallback_is_valid_model(self):
        """Minimax fallback should be a valid Model enum value."""
        fallback = FALLBACK_CHAIN[Model.MINIMAX_TEXT_01]
        assert isinstance(fallback, Model)


class TestZaiGlmFallbackChain:
    """Test Z.ai GLM has fallback chain entry."""

    def test_zai_glm_in_fallback_chain(self):
        """GLM_4 must have a fallback chain entry."""
        assert Model.GLM_4 in FALLBACK_CHAIN

    def test_zai_glm_fallback_is_different_provider(self):
        """Z.ai GLM fallback should be from a different provider."""
        fallback = FALLBACK_CHAIN[Model.GLM_4]
        glm_provider = get_provider(Model.GLM_4)
        fallback_provider = get_provider(fallback)
        assert (
            glm_provider != fallback_provider
        ), f"GLM-4 fallback should be cross-provider, got {fallback_provider}"

    def test_zai_glm_fallback_is_valid_model(self):
        """Z.ai GLM fallback should be a valid Model enum value."""
        fallback = FALLBACK_CHAIN[Model.GLM_4]
        assert isinstance(fallback, Model)


class TestProfileBuilding:
    """Test both new models are included in default profiles."""

    def test_minimax_in_default_profiles(self):
        """build_default_profiles() must include MINIMAX_TEXT_01."""
        profiles = build_default_profiles()
        assert Model.MINIMAX_TEXT_01 in profiles

    def test_minimax_profile_has_correct_provider(self):
        """Minimax profile should have 'minimax' provider."""
        profiles = build_default_profiles()
        assert profiles[Model.MINIMAX_TEXT_01].provider == "minimax"

    def test_minimax_profile_has_costs(self):
        """Minimax profile should have cost information."""
        profiles = build_default_profiles()
        profile = profiles[Model.MINIMAX_TEXT_01]
        assert profile.cost_per_1m_input > 0
        assert profile.cost_per_1m_output > 0

    def test_zai_glm_in_default_profiles(self):
        """build_default_profiles() must include GLM_4."""
        profiles = build_default_profiles()
        assert Model.GLM_4 in profiles

    def test_zai_glm_profile_has_correct_provider(self):
        """Z.ai GLM profile should have 'zhipu' provider."""
        profiles = build_default_profiles()
        assert profiles[Model.GLM_4].provider == "zhipu"

    def test_zai_glm_profile_has_costs(self):
        """Z.ai GLM profile should have cost information."""
        profiles = build_default_profiles()
        profile = profiles[Model.GLM_4]
        assert profile.cost_per_1m_input > 0
        assert profile.cost_per_1m_output > 0
