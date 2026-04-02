"""Unit tests for orchestrator.model_selector — TDD RED phase."""

from __future__ import annotations

import pytest

from orchestrator.model_selector import ModelSelector
from orchestrator.models import FALLBACK_CHAIN, Model, TaskType

# ── Helpers ───────────────────────────────────────────────────────────────────


def _all_healthy() -> dict[Model, bool]:
    return dict.fromkeys(Model, True)


def _all_unhealthy() -> dict[Model, bool]:
    return dict.fromkeys(Model, False)


def _health(**overrides: bool) -> dict[Model, bool]:
    h = _all_unhealthy()
    for model, status in overrides.items():
        h[Model[model]] = status
    return h


def _available_all(task_type: TaskType) -> list[Model]:
    return list(Model)


def _available_none(task_type: TaskType) -> list[Model]:
    return []


def _make_selector(health=None, available_fn=None) -> ModelSelector:
    return ModelSelector(
        api_health=health if health is not None else _all_healthy(),
        available_models_fn=available_fn or _available_all,
    )


# ── decomposition_model ───────────────────────────────────────────────────────


class TestDecompositionModel:
    def test_returns_model_instance(self):
        sel = _make_selector()
        result = sel.decomposition_model("Build a simple REST API")
        assert isinstance(result, Model)

    def test_prefers_qwen_when_healthy(self):
        sel = _make_selector(health=_all_healthy())
        result = sel.decomposition_model("Build a simple REST API")
        assert result == Model.QWEN_3_CODER_NEXT

    def test_skips_qwen_when_unhealthy(self):
        h = _all_healthy()
        h[Model.QWEN_3_CODER_NEXT] = False
        sel = _make_selector(health=h)
        result = sel.decomposition_model("Build a simple REST API")
        assert result != Model.QWEN_3_CODER_NEXT

    def test_falls_back_to_mimo_when_qwen_unhealthy(self):
        h = _all_healthy()
        h[Model.QWEN_3_CODER_NEXT] = False
        sel = _make_selector(health=h)
        result = sel.decomposition_model("project")
        assert result == Model.XIAOMI_MIMO_V2_FLASH

    def test_falls_back_to_gemini_flash_when_first_two_unhealthy(self):
        h = _all_healthy()
        h[Model.QWEN_3_CODER_NEXT] = False
        h[Model.XIAOMI_MIMO_V2_FLASH] = False
        sel = _make_selector(health=h)
        result = sel.decomposition_model("project")
        assert result == Model.GEMINI_FLASH

    def test_last_resort_stepfun(self):
        h = _all_healthy()
        h[Model.QWEN_3_CODER_NEXT] = False
        h[Model.XIAOMI_MIMO_V2_FLASH] = False
        h[Model.GEMINI_FLASH] = False
        sel = _make_selector(health=h)
        result = sel.decomposition_model("project")
        assert result == Model.STEPFUN_STEP_3_5_FLASH


# ── reviewer ──────────────────────────────────────────────────────────────────


class TestReviewer:
    def test_returns_none_when_no_candidates(self):
        sel = _make_selector(available_fn=_available_none)
        result = sel.reviewer(Model.GPT_4O, TaskType.CODE_GEN)
        assert result is None

    def test_returns_none_when_all_unhealthy(self):
        sel = _make_selector(health=_all_unhealthy())
        result = sel.reviewer(Model.GPT_4O, TaskType.CODE_GEN)
        assert result is None

    def test_returns_different_model_from_generator(self):
        sel = _make_selector()
        result = sel.reviewer(Model.GPT_4O, TaskType.CODE_GEN)
        assert result != Model.GPT_4O

    def test_prefers_cross_provider(self):
        # All models route via OpenRouter so get_provider() returns the same
        # value for all. The reviewer should still pick a *different* model.
        sel = _make_selector()
        generator = Model.GPT_4O
        result = sel.reviewer(generator, TaskType.CODE_GEN)
        assert result != generator


# ── fallback ──────────────────────────────────────────────────────────────────


class TestFallback:
    def test_returns_none_when_all_unhealthy(self):
        sel = _make_selector(health=_all_unhealthy())
        result = sel.fallback(Model.GPT_4O)
        assert result is None

    def test_uses_fallback_chain(self):
        # Find a model that has a defined fallback
        model_with_fallback = next(
            (m for m in FALLBACK_CHAIN if FALLBACK_CHAIN[m] is not None), None
        )
        if model_with_fallback is None:
            pytest.skip("No FALLBACK_CHAIN entries to test")

        expected = FALLBACK_CHAIN[model_with_fallback]
        h = _all_unhealthy()
        h[expected] = True
        sel = _make_selector(health=h)
        result = sel.fallback(model_with_fallback)
        assert result == expected

    def test_scans_all_models_when_chain_miss(self):
        # Use a model not in FALLBACK_CHAIN
        candidate = Model.GEMINI_FLASH
        h = _all_unhealthy()
        h[candidate] = True
        sel = _make_selector(health=h)
        # Any failed model not in FALLBACK_CHAIN should still find candidate
        failed = next(m for m in Model if m not in FALLBACK_CHAIN and m != candidate)
        result = sel.fallback(failed)
        assert result == candidate

    def test_does_not_return_same_model(self):
        h = _all_healthy()
        sel = _make_selector(health=h)
        result = sel.fallback(Model.GPT_4O)
        assert result != Model.GPT_4O


# ── next_tier ─────────────────────────────────────────────────────────────────


class TestNextTier:
    def test_returns_none_when_all_unhealthy(self):
        sel = _make_selector(health=_all_unhealthy())
        result = sel.next_tier(Model.GEMINI_FLASH_LITE, TaskType.CODE_GEN)
        assert result is None

    def test_escalates_from_cheap_tier(self):
        sel = _make_selector()
        result = sel.next_tier(Model.GEMINI_FLASH_LITE, TaskType.CODE_GEN)
        # Should return a tier-1 or tier-2 model
        assert result is not None
        assert result != Model.GEMINI_FLASH_LITE

    def test_returns_none_when_already_premium(self):
        # Premium models: GPT_4O, DEEPSEEK_REASONER, GEMINI_PRO
        # All premium are at max tier — next_tier should return None for them
        h = _all_healthy()
        sel = _make_selector(health=h)
        result = sel.next_tier(Model.GPT_4O, TaskType.CODE_GEN)
        assert result is None

    def test_returns_model_from_higher_tier(self):
        sel = _make_selector()
        cheap = Model.GPT_4O_MINI  # tier 0
        result = sel.next_tier(cheap, TaskType.CODE_GEN)
        assert result is not None
        # Must not be another tier-0 model
        tier_0 = {Model.GEMINI_FLASH_LITE, Model.GPT_4O_MINI}
        assert result not in tier_0

    def test_returns_none_when_no_available_models(self):
        sel = _make_selector(available_fn=_available_none)
        result = sel.next_tier(Model.GEMINI_FLASH_LITE, TaskType.CODE_GEN)
        assert result is None
