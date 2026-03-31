"""Tests for ModelRouting."""
from __future__ import annotations
import pytest


class TestModelTier:
    def test_three_tiers_exist(self):
        from orchestrator.model_routing import ModelTier
        assert ModelTier.PREMIUM is not None
        assert ModelTier.STANDARD is not None
        assert ModelTier.ECONOMY is not None

    def test_tiers_are_distinct(self):
        from orchestrator.model_routing import ModelTier
        assert ModelTier.PREMIUM != ModelTier.STANDARD
        assert ModelTier.STANDARD != ModelTier.ECONOMY


class TestTierRouting:
    def test_all_tiers_have_models(self):
        from orchestrator.model_routing import TIER_ROUTING, ModelTier
        for tier in ModelTier:
            assert tier in TIER_ROUTING
            assert len(TIER_ROUTING[tier]) > 0

    def test_premium_has_gpt4o(self):
        from orchestrator.model_routing import TIER_ROUTING, ModelTier

        # model_routing uses OpenRouter-prefixed names (e.g. "openai/gpt-4o")
        assert "openai/gpt-4o" in TIER_ROUTING[ModelTier.PREMIUM]

    def test_economy_has_cheap_model(self):
        from orchestrator.model_routing import TIER_ROUTING, ModelTier

        # Economy tier should not contain premium models
        assert "openai/gpt-4o" not in TIER_ROUTING[ModelTier.ECONOMY]


class TestSelectModel:
    def test_returns_first_model_when_no_preference(self):
        from orchestrator.model_routing import select_model, ModelTier, TIER_ROUTING
        result = select_model(ModelTier.PREMIUM)
        assert result == TIER_ROUTING[ModelTier.PREMIUM][0]

    def test_returns_preferred_if_in_tier(self):
        from orchestrator.model_routing import select_model, ModelTier

        # model_routing uses OpenRouter-prefixed names
        result = select_model(ModelTier.STANDARD, preferred="openai/gpt-4o-mini")
        assert result == "openai/gpt-4o-mini"

    def test_falls_back_to_first_if_preferred_not_in_tier(self):
        from orchestrator.model_routing import select_model, ModelTier, TIER_ROUTING
        result = select_model(ModelTier.ECONOMY, preferred="nonexistent-model")
        assert result == TIER_ROUTING[ModelTier.ECONOMY][0]


class TestGetTierForPhase:
    def test_reasoning_is_premium(self):
        from orchestrator.model_routing import get_tier_for_phase, ModelTier
        assert get_tier_for_phase("reasoning") == ModelTier.PREMIUM

    def test_unknown_phase_returns_standard(self):
        from orchestrator.model_routing import get_tier_for_phase, ModelTier
        assert get_tier_for_phase("unknown_phase_xyz") == ModelTier.STANDARD

    def test_summarize_is_economy(self):
        from orchestrator.model_routing import get_tier_for_phase, ModelTier
        assert get_tier_for_phase("summarize") == ModelTier.ECONOMY
