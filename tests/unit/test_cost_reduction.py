"""
Cost-reduction enforcement suite — proves the workflow USES its cost levers.

Goal: lock in the token-minimization / caching / cheap-routing behaviour so it
cannot silently regress. Each test asserts a concrete cost-saving guarantee of
the live execution path (not just that a helper class exists).

Levers covered:
  1. Response cache hit  → cost_usd == 0 AND no provider dispatch (no spend).
  2. Cache write-through  → a completed call is stored for reuse.
  3. VerificationGate veto → LLM never called when deterministic checks fail
                             (ENH-1 token save).
  4. CompletionJudge      → routes to the *cheapest* non-generator candidate
                             (ENH-2 cheap-tier maker-checker).
  5. Evaluator gate-veto  → no eval LLM spend on demonstrably broken artifacts.
  6. Default flags        → always-on cost features stay enabled by default.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.models import Model

# ──────────────────────────────────────────────────────────────────────────────
# Lever 1 + 2: response cache — hit is free and skips the provider entirely
# ──────────────────────────────────────────────────────────────────────────────


class _FakeCache:
    """Minimal DiskCache stand-in recording get/put and returning a fixed hit."""

    def __init__(self, hit: dict | None = None):
        self._hit = hit
        self.put_calls: list[tuple] = []
        self.get_calls: list[tuple] = []

    async def get(self, model_id, prompt, max_tokens, system, temperature):
        self.get_calls.append((model_id, prompt, max_tokens, system, temperature))
        return self._hit

    async def put(self, *args, **kwargs):
        self.put_calls.append((args, kwargs))


def _client_with_cache(cache):
    from orchestrator.infrastructure.llm_client import UnifiedClient

    # Construct without touching network clients.
    with patch.object(UnifiedClient, "_init_clients", lambda self: None):
        return UnifiedClient(cache=cache)


class TestResponseCacheSavesSpend:
    @pytest.mark.asyncio
    async def test_cache_hit_costs_zero_and_skips_dispatch(self):
        hit = {"response": "cached answer", "tokens_input": 100, "tokens_output": 50}
        cache = _FakeCache(hit=hit)
        client = _client_with_cache(cache)

        # _dispatch must NOT be called on a cache hit.
        client._dispatch = AsyncMock(side_effect=AssertionError("dispatch on cache hit!"))

        resp = await client.call(Model.GPT_4O, "prompt", max_tokens=500)

        assert resp.cached is True
        assert resp.cost_usd == 0.0, "cache hit must cost nothing"
        assert resp.text == "cached answer"
        client._dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_writes_through_for_reuse(self):
        cache = _FakeCache(hit=None)  # miss
        client = _client_with_cache(cache)

        fake_resp = MagicMock()
        fake_resp.text = "fresh"
        fake_resp.input_tokens = 10
        fake_resp.output_tokens = 5
        fake_resp.cost_usd = 0.002
        fake_resp.latency_ms = 0.0
        client._dispatch = AsyncMock(return_value=fake_resp)

        await client.call(Model.GPT_4O, "prompt", max_tokens=500, retries=0)

        assert len(cache.put_calls) == 1, "completed call must be cached for reuse"


# ──────────────────────────────────────────────────────────────────────────────
# Lever 3 + 5: VerificationGate veto — broken artifact never reaches the LLM
# ──────────────────────────────────────────────────────────────────────────────


class TestGateVetoSavesEvalSpend:
    @pytest.mark.asyncio
    async def test_failing_gate_makes_zero_llm_calls(self):
        from orchestrator.application.verification_gate import (
            VerificationCheck,
            VerificationGate,
        )
        from orchestrator.application.evaluator import EvaluatorService

        async def _fail(artifact: str):
            return False, "tests failed"

        gate = VerificationGate(checks=[VerificationCheck(name="t", run=_fail)])

        client = MagicMock()
        client.call = AsyncMock(side_effect=AssertionError("LLM called despite gate veto"))
        budget = AsyncMock()

        ev = EvaluatorService(
            client=client,
            budget=budget,
            get_models_fn=lambda t: [Model.GPT_4O],
            verification_gate=gate,
        )

        task = MagicMock()
        task.id = "t1"
        task.prompt = "do x"
        task.acceptance_threshold = 0.7
        task.type = MagicMock(value="code")

        report = await ev.evaluate(task, "broken artifact")

        assert report.passed_validators is False
        client.call.assert_not_called()
        budget.charge.assert_not_called()  # zero spend on a vetoed artifact


# ──────────────────────────────────────────────────────────────────────────────
# Lever 4: CompletionJudge routes to the cheapest non-generator candidate
# ──────────────────────────────────────────────────────────────────────────────


class TestJudgeUsesCheapTier:
    def test_from_models_picks_first_cheap_candidate(self):
        from orchestrator.services.completion_judge import CompletionJudge

        def _m(v):
            m = MagicMock()
            m.value = v
            return m

        generator = _m("openai/gpt-5")  # expensive generator
        cheap = _m("openai/gpt-4o-mini")  # cheap judge (ordered first)
        mid = _m("openai/gpt-4o")

        judge = CompletionJudge.from_models(
            client=MagicMock(),
            judge_candidates=[cheap, mid],  # cheapest-first ordering
            generator_model=generator,
        )
        assert judge is not None
        assert judge._judge_model is cheap, "judge must use the cheapest candidate"
        assert judge._judge_model is not generator, "judge must differ from generator"


# ──────────────────────────────────────────────────────────────────────────────
# Lever 6: always-on cost flags remain enabled by default
# ──────────────────────────────────────────────────────────────────────────────


class TestCostFlagsDefaultOn:
    def test_core_cost_features_enabled_by_default(self):
        from orchestrator.crosscutting.config import FeatureFlags

        flags = FeatureFlags()
        assert flags.cost_optimization_enabled is True
        assert flags.cache_optimizer_enabled is True
        assert flags.token_optimizer_enabled is True

    def test_cache_ttl_is_positive(self):
        from orchestrator.crosscutting.config import OrchestratorSettings

        settings = OrchestratorSettings()
        assert settings.cache_ttl_hours > 0, "response cache must have a live TTL"
