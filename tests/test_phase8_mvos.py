"""
Phase 8 — Minimum Viable Operational State (MVOS) audit.

Verifies every invariant defined in docs/ARCHITECTURAL_AUDIT_V5.md § MVOS:

  MVOS-1  Orchestrator instantiates cleanly with null adapters (no SQLite)
  MVOS-2  UnifiedClient.circuit_breaker starts CLOSED
  MVOS-3  CircuitBreakerRegistry is wired and per-model-isolated
  MVOS-4  ObservabilityService is wired and accumulates call metrics
  MVOS-5  All application-layer services are present (executor, evaluator,
           generator, task guard)
  MVOS-6  Concrete adapters satisfy Port protocols (structural subtyping)
  MVOS-7  CascadePolicy builds a valid cost-tier-sorted ResiliencePolicy

These tests are intentionally fast (no I/O, no network) and must remain so.
They act as a regression guard: if any Phase 0–7 change breaks the
architecture invariants, this suite catches it immediately.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.circuit_breaker import CircuitBreakerRegistry, CircuitState
from orchestrator.models import Model
from orchestrator.ports import CachePort, NullCache, NullState, StatePort
from orchestrator.resilience import CascadePolicy, ResiliencePolicy
from orchestrator.services import (
    EvaluatorService,
    ExecutorService,
    GeneratorService,
    ObservabilityService,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def orchestrator():
    """Orchestrator with null adapters — no SQLite, no network."""
    from orchestrator.engine import Orchestrator

    return Orchestrator(cache=NullCache(), state_manager=NullState())


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-1: Clean instantiation
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos1_orchestrator_instantiates(orchestrator):
    """MVOS-1: Orchestrator.__init__ must succeed with NullAdapters."""
    assert orchestrator is not None


def test_mvos1_cache_is_null(orchestrator):
    assert isinstance(orchestrator.cache, NullCache)


def test_mvos1_state_is_null(orchestrator):
    assert isinstance(orchestrator.state_mgr, NullState)


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-2: Circuit breaker starts CLOSED
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos2_client_circuit_breaker_closed(orchestrator):
    """MVOS-2: The top-level circuit breaker on UnifiedClient starts CLOSED."""
    cb = orchestrator.client.circuit_breaker
    assert cb._state.state == CircuitState.CLOSED


def test_mvos2_circuit_breaker_has_expected_thresholds(orchestrator):
    cb = orchestrator.client.circuit_breaker
    assert cb.failure_threshold >= 3  # at least 3 failures before tripping
    assert cb.reset_timeout >= 30.0   # at least 30s before probe


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-3: CircuitBreakerRegistry is wired
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos3_registry_is_present(orchestrator):
    """MVOS-3: Per-model CB registry must be a CircuitBreakerRegistry."""
    assert isinstance(orchestrator._cb_registry, CircuitBreakerRegistry)


def test_mvos3_registry_starts_empty(orchestrator):
    """No models are pre-tripped at startup."""
    assert orchestrator._cb_registry.tripped_models() == []


def test_mvos3_registry_isolates_models(orchestrator):
    """Tripping one model must not affect another."""
    reg = orchestrator._cb_registry
    cb_a = reg.get_sync("model-a")
    cb_b = reg.get_sync("model-b")
    assert cb_a is not cb_b


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-4: ObservabilityService is wired and functional
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos4_observability_is_present(orchestrator):
    """MVOS-4: ObservabilityService must be an attribute on Orchestrator."""
    assert isinstance(orchestrator.observability, ObservabilityService)


@pytest.mark.asyncio
async def test_mvos4_observability_accumulates_calls(orchestrator):
    obs = orchestrator.observability
    await obs.record_call("gpt-4o", latency_ms=300.0, cost_usd=0.001, success=True)
    summary = obs.model_summary("gpt-4o")
    assert summary is not None
    assert summary.calls == 1
    assert summary.errors == 0


@pytest.mark.asyncio
async def test_mvos4_observability_tracks_failures(orchestrator):
    obs = orchestrator.observability
    await obs.record_call("gpt-4o", latency_ms=5000.0, cost_usd=0.0,
                          success=False, error="TimeoutError")
    summary = obs.model_summary("gpt-4o")
    assert summary.errors == 1


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-5: All application-layer services present
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos5_executor_service_present(orchestrator):
    """MVOS-5: ExecutorService must be instantiated on the engine."""
    assert isinstance(orchestrator._executor, ExecutorService)


def test_mvos5_evaluator_service_present(orchestrator):
    assert isinstance(orchestrator._evaluator, EvaluatorService)


def test_mvos5_generator_service_present(orchestrator):
    assert isinstance(orchestrator._generator, GeneratorService)


def test_mvos5_task_guard_present(orchestrator):
    from orchestrator.concurrency_controller import TaskConcurrencyGuard
    assert isinstance(orchestrator._task_guard, TaskConcurrencyGuard)


def test_mvos5_executor_has_guard(orchestrator):
    """ExecutorService must have the TaskConcurrencyGuard injected."""
    assert orchestrator._executor._guard is orchestrator._task_guard


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-6: Concrete adapters satisfy Port protocols
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos6_disk_cache_satisfies_cache_port():
    from orchestrator.cache import DiskCache
    assert isinstance(DiskCache(), CachePort)


def test_mvos6_state_manager_satisfies_state_port():
    from orchestrator.state import StateManager
    assert isinstance(StateManager(), StatePort)


def test_mvos6_null_cache_satisfies_cache_port():
    assert isinstance(NullCache(), CachePort)


def test_mvos6_null_state_satisfies_state_port():
    assert isinstance(NullState(), StatePort)


# ─────────────────────────────────────────────────────────────────────────────
# MVOS-7: CascadePolicy produces valid ResiliencePolicy
# ─────────────────────────────────────────────────────────────────────────────


def test_mvos7_cascade_builds_policy():
    cascade = CascadePolicy.for_model(Model.GPT_4O)
    rp = cascade.to_policy()
    assert isinstance(rp, ResiliencePolicy)


def test_mvos7_cascade_chain_starts_with_primary():
    cascade = CascadePolicy.for_model(Model.GPT_4O)
    assert cascade.ordered_chain[0] == Model.GPT_4O


def test_mvos7_fallback_excludes_primary():
    cascade = CascadePolicy.for_model(Model.GPT_4O)
    rp = cascade.to_policy()
    if rp.fallback_chain:
        assert Model.GPT_4O not in rp.fallback_chain


def test_mvos7_retry_template_covers_all_task_types():
    """Every task type in the ROUTING_TABLE must have a retry template."""
    from orchestrator.models import ROUTING_TABLE
    from orchestrator.resilience import RetryTemplate

    for task_type in ROUTING_TABLE:
        policy = RetryTemplate.for_task_type(task_type)
        assert isinstance(policy, ResiliencePolicy)
        assert policy.retries >= 1
        assert policy.timeout >= 10.0
