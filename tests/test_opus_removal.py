"""
Test suite for Claude Opus removal and routing optimization.

Tests verify that:
- Claude Opus is completely removed from the model infrastructure
- All routing tables have been updated with correct replacements
- Fallback chains are consistent and cross-provider
- No dangling references exist
"""

import pytest
from orchestrator.models import (
    Model,
    TaskType,
    COST_TABLE,
    ROUTING_TABLE,
    FALLBACK_CHAIN,
)


class TestClaudeOpusRemovalFromEnum:
    """Verify Claude Opus is not in the Model enum."""

    def test_claude_opus_not_in_model_enum(self):
        """Claude Opus should be completely removed from Model enum."""
        model_values = [m.value for m in Model]
        assert "claude-opus-4-6" not in model_values, \
            "CLAUDE_OPUS must be removed from Model enum"

    def test_all_models_are_valid_enum_members(self):
        """Verify all models in enum can be accessed."""
        models = [m for m in Model]
        assert len(models) > 0, "Model enum should have at least one model"
        # Should not raise AttributeError
        assert hasattr(Model, 'DEEPSEEK_CHAT')
        assert hasattr(Model, 'MINIMAX_3')
        assert hasattr(Model, 'ZAI_GLM')


class TestClaudeOpusRemovalFromCostTable:
    """Verify Claude Opus costs are removed."""

    def test_claude_opus_not_in_cost_table(self):
        """CLAUDE_OPUS should not have cost entries."""
        # Try to access Model.CLAUDE_OPUS - should fail at import if enum removed
        try:
            opus_model = Model.CLAUDE_OPUS
            assert opus_model not in COST_TABLE, \
                "CLAUDE_OPUS should not be in COST_TABLE"
        except AttributeError:
            # Expected - CLAUDE_OPUS doesn't exist in enum
            pass

    def test_all_routing_models_have_costs(self):
        """Every model in ROUTING_TABLE must have a cost entry."""
        models_in_routing = set()
        for task_type, model_list in ROUTING_TABLE.items():
            for model in model_list:
                models_in_routing.add(model)

        for model in models_in_routing:
            assert model in COST_TABLE, \
                f"Model {model} in ROUTING_TABLE but not in COST_TABLE"


class TestRoutingTableUpdates:
    """Verify ROUTING_TABLE has been updated correctly."""

    def test_code_review_no_opus(self):
        """CODE_REVIEW should not contain CLAUDE_OPUS."""
        try:
            opus = Model.CLAUDE_OPUS
            assert opus not in ROUTING_TABLE[TaskType.CODE_REVIEW], \
                "CLAUDE_OPUS should be removed from CODE_REVIEW routing"
        except AttributeError:
            pass  # Expected

    def test_code_review_has_minimax(self):
        """CODE_REVIEW should include MINIMAX_3 (replacement for Opus)."""
        assert Model.MINIMAX_3 in ROUTING_TABLE[TaskType.CODE_REVIEW], \
            "MINIMAX_3 should be in CODE_REVIEW routing"

    def test_reasoning_no_opus(self):
        """REASONING should not contain CLAUDE_OPUS."""
        try:
            opus = Model.CLAUDE_OPUS
            assert opus not in ROUTING_TABLE[TaskType.REASONING], \
                "CLAUDE_OPUS should be removed from REASONING routing"
        except AttributeError:
            pass

    def test_reasoning_has_sonnet(self):
        """REASONING should include CLAUDE_SONNET (replacement for Opus escalation)."""
        assert Model.CLAUDE_SONNET in ROUTING_TABLE[TaskType.REASONING], \
            "CLAUDE_SONNET should be in REASONING routing for quality escalation"

    def test_writing_no_opus(self):
        """WRITING should not contain CLAUDE_OPUS (was primary)."""
        try:
            opus = Model.CLAUDE_OPUS
            assert opus not in ROUTING_TABLE[TaskType.WRITING], \
                "CLAUDE_OPUS should be removed from WRITING routing"
        except AttributeError:
            pass

    def test_writing_has_zai_glm_first(self):
        """WRITING should have ZAI_GLM as primary replacement."""
        assert Model.ZAI_GLM in ROUTING_TABLE[TaskType.WRITING], \
            "ZAI_GLM should be in WRITING routing"
        # Should be first or early in the list
        writing_models = ROUTING_TABLE[TaskType.WRITING]
        zai_index = writing_models.index(Model.ZAI_GLM)
        assert zai_index <= 1, \
            "ZAI_GLM should be primary (#1) or early secondary (#2) in WRITING"

    def test_writing_has_sonnet_escalation(self):
        """WRITING should have CLAUDE_SONNET for quality escalation."""
        assert Model.CLAUDE_SONNET in ROUTING_TABLE[TaskType.WRITING], \
            "CLAUDE_SONNET should be escalation tier in WRITING"

    def test_evaluate_no_opus(self):
        """EVALUATE should not contain CLAUDE_OPUS."""
        try:
            opus = Model.CLAUDE_OPUS
            assert opus not in ROUTING_TABLE[TaskType.EVALUATE], \
                "CLAUDE_OPUS should be removed from EVALUATE routing"
        except AttributeError:
            pass

    def test_evaluate_has_minimax(self):
        """EVALUATE should include MINIMAX_3 (replacement for Opus)."""
        assert Model.MINIMAX_3 in ROUTING_TABLE[TaskType.EVALUATE], \
            "MINIMAX_3 should be in EVALUATE routing"

    def test_all_task_types_have_primary_model(self):
        """Every task type must have at least one primary model."""
        for task_type, model_list in ROUTING_TABLE.items():
            assert len(model_list) > 0, \
                f"{task_type} has no routing models"
            assert isinstance(model_list[0], Model), \
                f"{task_type} first model is not a Model enum"

    def test_all_routing_models_exist(self):
        """Every model in ROUTING_TABLE must be a valid Model enum member."""
        for task_type, model_list in ROUTING_TABLE.items():
            for model in model_list:
                assert isinstance(model, Model), \
                    f"{task_type} contains invalid model: {model}"


class TestFallbackChainUpdates:
    """Verify FALLBACK_CHAIN has been updated correctly."""

    def test_claude_opus_not_in_fallback_chain_as_key(self):
        """CLAUDE_OPUS should not be a key in FALLBACK_CHAIN."""
        try:
            opus = Model.CLAUDE_OPUS
            assert opus not in FALLBACK_CHAIN, \
                "CLAUDE_OPUS should not be a key in FALLBACK_CHAIN"
        except AttributeError:
            pass

    def test_deepseek_reasoner_not_fallback_to_opus(self):
        """DEEPSEEK_REASONER should not fallback to CLAUDE_OPUS."""
        try:
            opus = Model.CLAUDE_OPUS
            reasoner_fallback = FALLBACK_CHAIN.get(Model.DEEPSEEK_REASONER)
            assert reasoner_fallback != opus, \
                "DEEPSEEK_REASONER should not fallback to CLAUDE_OPUS"
        except AttributeError:
            pass

    def test_deepseek_reasoner_fallback_to_sonnet(self):
        """DEEPSEEK_REASONER should fallback to CLAUDE_SONNET (cross-provider)."""
        reasoner_fallback = FALLBACK_CHAIN.get(Model.DEEPSEEK_REASONER)
        assert reasoner_fallback == Model.CLAUDE_SONNET, \
            "DEEPSEEK_REASONER should fallback to CLAUDE_SONNET"

    def test_all_fallback_targets_exist(self):
        """Every fallback target must be a valid Model enum member."""
        for source_model, target_model in FALLBACK_CHAIN.items():
            assert isinstance(target_model, Model), \
                f"Fallback target for {source_model} is not a Model: {target_model}"
            # Target should exist in enum
            try:
                Model[target_model.name]
            except KeyError:
                pytest.fail(f"Fallback target {target_model} not in Model enum")

    def test_no_self_fallback_chains(self):
        """No model should fallback to itself."""
        for source_model, target_model in FALLBACK_CHAIN.items():
            assert source_model != target_model, \
                f"{source_model} has self-fallback (invalid circular chain)"

    def test_fallback_chain_cross_provider(self):
        """Critical fallback chains should be cross-provider."""
        from orchestrator.models import get_provider

        critical_chains = [
            (Model.DEEPSEEK_REASONER, Model.CLAUDE_SONNET),
            (Model.DEEPSEEK_CHAT, Model.CLAUDE_SONNET),
        ]

        for source, target in critical_chains:
            source_provider = get_provider(source)
            target_provider = get_provider(target)
            assert source_provider != target_provider, \
                f"{source} → {target} is not cross-provider " \
                f"({source_provider} → {target_provider})"


class TestModelConsistency:
    """Verify overall consistency after Opus removal."""

    def test_no_broken_routing_references(self):
        """All models in ROUTING_TABLE must be in COST_TABLE."""
        models_in_routing = set()
        for model_list in ROUTING_TABLE.values():
            models_in_routing.update(model_list)

        for model in models_in_routing:
            assert model in COST_TABLE, \
                f"{model} is in ROUTING_TABLE but not in COST_TABLE"

    def test_cost_table_completeness(self):
        """COST_TABLE entries must have both input and output costs."""
        for model, costs in COST_TABLE.items():
            assert "input" in costs, \
                f"{model} missing input cost"
            assert "output" in costs, \
                f"{model} missing output cost"
            assert costs["input"] > 0, \
                f"{model} has invalid input cost: {costs['input']}"
            assert costs["output"] > 0, \
                f"{model} has invalid output cost: {costs['output']}"

    def test_minimax_cheaper_than_opus_was(self):
        """MINIMAX_3 should be significantly cheaper than Claude Opus was."""
        minimax_cost = COST_TABLE[Model.MINIMAX_3]
        # Opus was: input=$15, output=$75
        # Minimax should be much cheaper
        assert minimax_cost["input"] < 5.0, \
            "MINIMAX_3 input cost should be much lower than Opus's $15"
        assert minimax_cost["output"] < 10.0, \
            "MINIMAX_3 output cost should be much lower than Opus's $75"

    def test_zai_glm_cheaper_than_opus_was(self):
        """ZAI_GLM should be significantly cheaper than Claude Opus was."""
        zai_cost = COST_TABLE[Model.ZAI_GLM]
        # Opus was: input=$15, output=$75
        # ZAI GLM should be much cheaper
        assert zai_cost["input"] < 5.0, \
            "ZAI_GLM input cost should be much lower than Opus's $15"
        assert zai_cost["output"] < 20.0, \
            "ZAI_GLM output cost should be much lower than Opus's $75"

    def test_routing_model_count_reasonable(self):
        """Each task type should have 3-5 models for good fallback coverage."""
        for task_type, model_list in ROUTING_TABLE.items():
            assert 3 <= len(model_list) <= 6, \
                f"{task_type} has {len(model_list)} models (expected 3-5)"
