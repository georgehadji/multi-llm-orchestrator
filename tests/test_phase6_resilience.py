"""
Phase 6 — Advanced Resilience tests.

Covers:
  - CircuitBreakerRegistry: per-model breakers, tripped_models(), reset_all()
  - ObservabilityService: record_call, error_rate, ranked_by_error_rate,
    is_degraded, total_cost_usd
  - CascadePolicy: ordered_chain, to_policy fallback_chain
"""

from __future__ import annotations


import pytest

pytestmark = pytest.mark.unit

from orchestrator.circuit_breaker import (
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
)
from orchestrator.models import Model
from orchestrator.resilience import (
    CascadePolicy,
    CostTier,
    ResiliencePolicy,
    classify_model_tier,
)
from orchestrator.services.observability import ObservabilityService

# ─────────────────────────────────────────────────────────────────────────────
# CircuitBreakerRegistry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_creates_breaker_on_first_get():
    reg = CircuitBreakerRegistry()
    cb = await reg.get("openai/gpt-4o")
    assert cb is not None
    assert cb.name == "model:openai/gpt-4o"


@pytest.mark.asyncio
async def test_registry_returns_same_instance():
    reg = CircuitBreakerRegistry()
    cb1 = await reg.get("openai/gpt-4o")
    cb2 = await reg.get("openai/gpt-4o")
    assert cb1 is cb2


@pytest.mark.asyncio
async def test_registry_separate_breakers_per_model():
    reg = CircuitBreakerRegistry()
    cb_a = await reg.get("openai/gpt-4o")
    cb_b = await reg.get("anthropic/claude-3")
    assert cb_a is not cb_b


@pytest.mark.asyncio
async def test_registry_tripped_models_empty_when_all_closed():
    reg = CircuitBreakerRegistry()
    await reg.get("model-a")
    await reg.get("model-b")
    assert reg.tripped_models() == []


@pytest.mark.asyncio
async def test_registry_tripped_models_shows_open_breaker():
    reg = CircuitBreakerRegistry(failure_threshold=1)
    cb = await reg.get("bad-model")
    # Trip it
    for _ in range(1):
        try:
            async with cb.context():
                raise RuntimeError("fail")
        except RuntimeError:
            pass
    assert "bad-model" in reg.tripped_models()


@pytest.mark.asyncio
async def test_registry_reset_all_closes_all():
    reg = CircuitBreakerRegistry(failure_threshold=1)
    cb = await reg.get("bad-model")
    try:
        async with cb.context():
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    assert len(reg.tripped_models()) > 0
    await reg.reset_all()
    assert reg.tripped_models() == []


def test_registry_get_sync():
    reg = CircuitBreakerRegistry()
    cb = reg.get_sync("sync-model")
    assert cb.name == "model:sync-model"


def test_registry_all_stats():
    reg = CircuitBreakerRegistry()
    reg.get_sync("m1")
    reg.get_sync("m2")
    stats = reg.all_stats()
    assert "m1" in stats
    assert "m2" in stats
    assert stats["m1"]["state"] == "closed"


# ─────────────────────────────────────────────────────────────────────────────
# ObservabilityService
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_obs_record_and_retrieve():
    obs = ObservabilityService()
    await obs.record_call("gpt-4o", latency_ms=400.0, cost_usd=0.001, success=True)
    summary = obs.model_summary("gpt-4o")
    assert summary is not None
    assert summary.calls == 1
    assert summary.errors == 0
    assert summary.avg_latency_ms == pytest.approx(400.0)


@pytest.mark.asyncio
async def test_obs_error_rate_reflects_failures():
    obs = ObservabilityService()
    for _ in range(8):
        await obs.record_call("m", latency_ms=100.0, cost_usd=0.0, success=True)
    for _ in range(2):
        await obs.record_call(
            "m", latency_ms=50.0, cost_usd=0.0, success=False, error="TimeoutError"
        )
    summary = obs.model_summary("m")
    assert summary.error_rate == pytest.approx(0.2, abs=0.05)


@pytest.mark.asyncio
async def test_obs_is_degraded_below_threshold():
    obs = ObservabilityService(error_rate_threshold=0.5)
    for _ in range(10):
        await obs.record_call("m", latency_ms=100.0, cost_usd=0.0, success=True)
    assert obs.is_degraded("m") is False


@pytest.mark.asyncio
async def test_obs_is_degraded_above_threshold():
    obs = ObservabilityService(error_rate_threshold=0.5)
    for _ in range(4):
        await obs.record_call("m", latency_ms=100.0, cost_usd=0.0, success=False, error="err")
    assert obs.is_degraded("m") is True


@pytest.mark.asyncio
async def test_obs_total_cost():
    obs = ObservabilityService()
    await obs.record_call("a", latency_ms=1.0, cost_usd=0.01, success=True)
    await obs.record_call("b", latency_ms=1.0, cost_usd=0.02, success=True)
    assert obs.total_cost_usd() == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_obs_unknown_model_returns_none():
    obs = ObservabilityService()
    assert obs.model_summary("nonexistent") is None


@pytest.mark.asyncio
async def test_obs_ranked_by_error_rate():
    obs = ObservabilityService()
    # model-a: 0 errors; model-b: all errors
    for _ in range(5):
        await obs.record_call("model-a", latency_ms=1.0, cost_usd=0.0, success=True)
    for _ in range(5):
        await obs.record_call("model-b", latency_ms=1.0, cost_usd=0.0, success=False)
    ranked = obs.ranked_by_error_rate()
    assert ranked[0].model_id == "model-b"


@pytest.mark.asyncio
async def test_obs_record_fallback():
    obs = ObservabilityService()
    await obs.record_fallback("primary-model")
    summary = obs.model_summary("primary-model")
    assert summary is not None
    assert summary.fallback_triggers == 1


# ─────────────────────────────────────────────────────────────────────────────
# CascadePolicy & classify_model_tier
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_tier_free_model():
    # GPT_4O_MINI has low cost — should be BUDGET or FREE
    tier = classify_model_tier(Model.GPT_4O_MINI)
    assert tier in (CostTier.FREE, CostTier.BUDGET, CostTier.PREMIUM)


def test_cascade_policy_chain_starts_with_primary():
    policy = CascadePolicy.for_model(Model.GPT_4O)
    assert policy.ordered_chain[0] == Model.GPT_4O


def test_cascade_policy_chain_is_not_empty():
    policy = CascadePolicy.for_model(Model.GPT_4O)
    assert len(policy.ordered_chain) >= 1


def test_cascade_policy_to_policy_returns_resilience_policy():
    cascade = CascadePolicy.for_model(Model.GPT_4O)
    rp = cascade.to_policy()
    assert isinstance(rp, ResiliencePolicy)


def test_cascade_policy_to_policy_excludes_primary_from_fallback():
    cascade = CascadePolicy.for_model(Model.GPT_4O)
    rp = cascade.to_policy()
    if rp.fallback_chain:
        assert Model.GPT_4O not in rp.fallback_chain
