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
        assert get_provider(Model.MINIMAX_3) == "minimax"

    def test_minimax_3_model_value_correct(self):
        """Verify MINIMAX_3 enum value is set correctly."""
        assert Model.MINIMAX_3.value == "minimax-3"


class TestZaiGlmProviderDetection:
    """Test get_provider() correctly identifies Z.ai/GLM models."""

    def test_zai_glm_returns_zhipu_provider(self):
        """Z.ai GLM-4 should be detected as 'zhipu' provider."""
        assert get_provider(Model.ZAI_GLM) == "zhipu"

    def test_zai_glm_model_value_correct(self):
        """Verify ZAI_GLM enum value is set correctly."""
        assert Model.ZAI_GLM.value == "zai-glm-4"


class TestMinimaxCostTable:
    """Test Minimax-3 has cost entries."""

    def test_minimax_3_in_cost_table(self):
        """MINIMAX_3 must have cost entries."""
        assert Model.MINIMAX_3 in COST_TABLE

    def test_minimax_3_has_input_cost(self):
        """MINIMAX_3 cost entry must include input cost."""
        assert "input" in COST_TABLE[Model.MINIMAX_3]

    def test_minimax_3_has_output_cost(self):
        """MINIMAX_3 cost entry must include output cost."""
        assert "output" in COST_TABLE[Model.MINIMAX_3]

    def test_minimax_3_costs_are_positive(self):
        """MINIMAX_3 costs must be positive numbers."""
        costs = COST_TABLE[Model.MINIMAX_3]
        assert costs["input"] > 0
        assert costs["output"] > 0


class TestZaiGlmCostTable:
    """Test Z.ai GLM-4 has cost entries."""

    def test_zai_glm_in_cost_table(self):
        """ZAI_GLM must have cost entries."""
        assert Model.ZAI_GLM in COST_TABLE

    def test_zai_glm_has_input_cost(self):
        """ZAI_GLM cost entry must include input cost."""
        assert "input" in COST_TABLE[Model.ZAI_GLM]

    def test_zai_glm_has_output_cost(self):
        """ZAI_GLM cost entry must include output cost."""
        assert "output" in COST_TABLE[Model.ZAI_GLM]

    def test_zai_glm_costs_are_positive(self):
        """ZAI_GLM costs must be positive numbers."""
        costs = COST_TABLE[Model.ZAI_GLM]
        assert costs["input"] > 0
        assert costs["output"] > 0


class TestMinimaxRoutingTable:
    """Test Minimax appears in routing table for appropriate task types."""

    def test_minimax_in_at_least_one_task_type(self):
        """Minimax should be routed for at least one task type."""
        found = False
        for task_type, model_list in ROUTING_TABLE.items():
            if Model.MINIMAX_3 in model_list:
                found = True
                break
        assert found, "MINIMAX_3 not found in any ROUTING_TABLE entry"

    def test_minimax_in_reasoning_task(self):
        """Minimax should be included for reasoning tasks (strength)."""
        assert Model.MINIMAX_3 in ROUTING_TABLE[TaskType.REASONING]

    def test_minimax_in_code_gen_task(self):
        """Minimax should be included for code generation (efficient reasoning)."""
        assert Model.MINIMAX_3 in ROUTING_TABLE[TaskType.CODE_GEN]


class TestZaiGlmRoutingTable:
    """Test Z.ai GLM appears in routing table for appropriate task types."""

    def test_zai_glm_in_at_least_one_task_type(self):
        """Z.ai GLM should be routed for at least one task type."""
        found = False
        for task_type, model_list in ROUTING_TABLE.items():
            if Model.ZAI_GLM in model_list:
                found = True
                break
        assert found, "ZAI_GLM not found in any ROUTING_TABLE entry"

    def test_zai_glm_in_writing_task(self):
        """Z.ai GLM should be included for writing tasks (general strength)."""
        assert Model.ZAI_GLM in ROUTING_TABLE[TaskType.WRITING]

    def test_zai_glm_in_code_gen_task(self):
        """Z.ai GLM should be included for code generation."""
        assert Model.ZAI_GLM in ROUTING_TABLE[TaskType.CODE_GEN]


class TestMinimaxFallbackChain:
    """Test Minimax has fallback chain entry."""

    def test_minimax_in_fallback_chain(self):
        """MINIMAX_3 must have a fallback chain entry."""
        assert Model.MINIMAX_3 in FALLBACK_CHAIN

    def test_minimax_fallback_is_different_provider(self):
        """Minimax fallback should be from a different provider."""
        fallback = FALLBACK_CHAIN[Model.MINIMAX_3]
        minimax_provider = get_provider(Model.MINIMAX_3)
        fallback_provider = get_provider(fallback)
        assert minimax_provider != fallback_provider, \
            f"Minimax fallback should be cross-provider, got {fallback_provider}"

    def test_minimax_fallback_is_valid_model(self):
        """Minimax fallback should be a valid Model enum value."""
        fallback = FALLBACK_CHAIN[Model.MINIMAX_3]
        assert isinstance(fallback, Model)


class TestZaiGlmFallbackChain:
    """Test Z.ai GLM has fallback chain entry."""

    def test_zai_glm_in_fallback_chain(self):
        """ZAI_GLM must have a fallback chain entry."""
        assert Model.ZAI_GLM in FALLBACK_CHAIN

    def test_zai_glm_fallback_is_different_provider(self):
        """Z.ai GLM fallback should be from a different provider."""
        fallback = FALLBACK_CHAIN[Model.ZAI_GLM]
        zai_provider = get_provider(Model.ZAI_GLM)
        fallback_provider = get_provider(fallback)
        assert zai_provider != fallback_provider, \
            f"Z.ai GLM fallback should be cross-provider, got {fallback_provider}"

    def test_zai_glm_fallback_is_valid_model(self):
        """Z.ai GLM fallback should be a valid Model enum value."""
        fallback = FALLBACK_CHAIN[Model.ZAI_GLM]
        assert isinstance(fallback, Model)


class TestProfileBuilding:
    """Test both new models are included in default profiles."""

    def test_minimax_in_default_profiles(self):
        """build_default_profiles() must include MINIMAX_3."""
        profiles = build_default_profiles()
        assert Model.MINIMAX_3 in profiles

    def test_minimax_profile_has_correct_provider(self):
        """Minimax profile should have 'minimax' provider."""
        profiles = build_default_profiles()
        assert profiles[Model.MINIMAX_3].provider == "minimax"

    def test_minimax_profile_has_costs(self):
        """Minimax profile should have cost information."""
        profiles = build_default_profiles()
        profile = profiles[Model.MINIMAX_3]
        assert profile.cost_per_1m_input > 0
        assert profile.cost_per_1m_output > 0

    def test_zai_glm_in_default_profiles(self):
        """build_default_profiles() must include ZAI_GLM."""
        profiles = build_default_profiles()
        assert Model.ZAI_GLM in profiles

    def test_zai_glm_profile_has_correct_provider(self):
        """Z.ai GLM profile should have 'zhipu' provider."""
        profiles = build_default_profiles()
        assert profiles[Model.ZAI_GLM].provider == "zhipu"

    def test_zai_glm_profile_has_costs(self):
        """Z.ai GLM profile should have cost information."""
        profiles = build_default_profiles()
        profile = profiles[Model.ZAI_GLM]
        assert profile.cost_per_1m_input > 0
        assert profile.cost_per_1m_output > 0
