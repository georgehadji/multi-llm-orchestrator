"""
Tests for TelemetryCollector — extended telemetry (Improvement 3).
Covers: real p95 via sorted buffer, cost EMA, validator_fail_count,
        error_rate(), record_policy_violation(), trust dynamics.
"""
from __future__ import annotations

import pytest
from orchestrator.models import Model, build_default_profiles
from orchestrator.telemetry import TelemetryCollector, _LATENCY_BUFFER_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def profiles():
    return build_default_profiles()


@pytest.fixture()
def collector(profiles):
    return TelemetryCollector(profiles)


@pytest.fixture()
def model():
    return Model.KIMI_K2_5


# ─────────────────────────────────────────────────────────────────────────────
# Basic call tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_call_count_increments(collector, profiles, model):
    collector.record_call(model, 1000.0, 0.001, success=True)
    assert profiles[model].call_count == 1
    collector.record_call(model, 1000.0, 0.001, success=True)
    assert profiles[model].call_count == 2


def test_failure_count_increments_on_failure(collector, profiles, model):
    collector.record_call(model, 0.0, 0.0, success=False)
    assert profiles[model].failure_count == 1


def test_failure_count_unchanged_on_success(collector, profiles, model):
    collector.record_call(model, 1000.0, 0.001, success=True)
    assert profiles[model].failure_count == 0


def test_unknown_model_is_ignored(collector):
    # Should not raise; just log a warning
    collector.record_call("nonexistent-model", 1000.0, 0.001, success=True)


# ─────────────────────────────────────────────────────────────────────────────
# Latency EMA
# ─────────────────────────────────────────────────────────────────────────────

def test_latency_ema_moves_toward_new_value(collector, profiles, model):
    initial = profiles[model].avg_latency_ms
    collector.record_call(model, 5000.0, 0.001, success=True)
    # EMA should be between initial and 5000
    new_val = profiles[model].avg_latency_ms
    assert initial < new_val < 5000.0


def test_zero_latency_does_not_update_ema(collector, profiles, model):
    initial = profiles[model].avg_latency_ms
    collector.record_call(model, 0.0, 0.001, success=True)
    assert profiles[model].avg_latency_ms == initial


# ─────────────────────────────────────────────────────────────────────────────
# Real p95 via sorted buffer
# ─────────────────────────────────────────────────────────────────────────────

def test_p95_is_not_double_avg_after_real_samples(collector, profiles, model):
    """Real p95 from sorted buffer must differ from the old 2*avg approximation."""
    for i in range(20):
        collector.record_call(model, float(i * 100 + 100), 0.001, success=True)
    p95 = profiles[model].latency_p95_ms
    avg = profiles[model].avg_latency_ms
    # Real p95 is not 2× avg (the old approximation); just verify they diverge
    assert p95 != pytest.approx(2 * avg, rel=0.01)


def test_latency_buffer_grows_with_samples(collector, profiles, model):
    for i in range(10):
        collector.record_call(model, float(i * 200 + 100), 0.001, success=True)
    assert len(profiles[model].latency_samples) == 10


def test_latency_buffer_max_50_samples(collector, profiles, model):
    """Feed 60 samples; buffer must not exceed _LATENCY_BUFFER_SIZE (50)."""
    for i in range(60):
        collector.record_call(model, float(i * 10 + 50), 0.001, success=True)
    assert len(profiles[model].latency_samples) == _LATENCY_BUFFER_SIZE


def test_latency_buffer_is_sorted_ascending(collector, profiles, model):
    """Sorted buffer must always be in non-decreasing order."""
    import random
    random.seed(42)
    for _ in range(30):
        collector.record_call(model, random.uniform(100, 5000), 0.001, success=True)
    samples = profiles[model].latency_samples
    assert samples == sorted(samples)


def test_p95_is_within_sample_range(collector, profiles, model):
    """p95 must be a value that exists in (or near the top of) the sample range."""
    for i in range(30):
        collector.record_call(model, float(i * 100 + 100), 0.001, success=True)
    p95 = profiles[model].latency_p95_ms
    samples = profiles[model].latency_samples
    assert p95 >= samples[0]  # must be at least the min
    assert p95 <= samples[-1]  # must be at most the max


def test_zero_latency_samples_not_added_to_buffer(collector, profiles, model):
    """Cache hits (latency_ms=0) must not pollute the latency buffer."""
    collector.record_call(model, 0.0, 0.0, success=True)
    assert len(profiles[model].latency_samples) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Cost EMA
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_ema_updates_on_positive_cost(collector, profiles, model):
    initial = profiles[model].avg_cost_usd  # 0.0
    collector.record_call(model, 1000.0, 0.001, success=True)
    # First seed: avg_cost_usd should become 0.001 (first observation)
    assert profiles[model].avg_cost_usd > 0.0


def test_cost_ema_zero_cost_is_ignored(collector, profiles, model):
    """cost_usd=0 (cache hits / failed calls) must not drag EMA toward zero."""
    # Seed with a known positive cost
    collector.record_call(model, 1000.0, 0.005, success=True)
    after_first = profiles[model].avg_cost_usd
    assert after_first > 0.0
    # Now record a zero-cost call
    collector.record_call(model, 0.0, 0.0, success=True)
    after_zero = profiles[model].avg_cost_usd
    # EMA must not have changed (skipped)
    assert after_zero == after_first


def test_cost_ema_converges_toward_observed_values(collector, profiles, model):
    """Many identical cost observations should move EMA toward that value."""
    target = 0.00042
    for _ in range(50):
        collector.record_call(model, 500.0, target, success=True)
    # After 50 EMA steps, should be close to target
    assert abs(profiles[model].avg_cost_usd - target) < target * 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Success rate rolling window
# ─────────────────────────────────────────────────────────────────────────────

def test_success_rate_100_percent_all_success(collector, profiles, model):
    for _ in range(5):
        collector.record_call(model, 500.0, 0.001, success=True)
    assert profiles[model].success_rate == pytest.approx(1.0)


def test_success_rate_50_percent_after_mixed(collector, profiles, model):
    for _ in range(5):
        collector.record_call(model, 500.0, 0.001, success=True)
    for _ in range(5):
        collector.record_call(model, 0.0, 0.0, success=False)
    assert profiles[model].success_rate == pytest.approx(0.5)


def test_success_rate_window_evicts_old_results(collector, profiles, model):
    """After 10 successes followed by 10 failures, only failures should be in window."""
    for _ in range(10):
        collector.record_call(model, 500.0, 0.001, success=True)
    for _ in range(10):
        collector.record_call(model, 0.0, 0.0, success=False)
    # Window = last 10 calls = 10 failures → success_rate should be 0
    assert profiles[model].success_rate == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Quality EMA
# ─────────────────────────────────────────────────────────────────────────────

def test_quality_ema_updated_when_score_provided(collector, profiles, model):
    initial = profiles[model].quality_score  # 0.8
    collector.record_call(model, 500.0, 0.001, success=True, quality_score=1.0)
    assert profiles[model].quality_score > initial


def test_quality_ema_unchanged_when_no_score(collector, profiles, model):
    initial = profiles[model].quality_score
    collector.record_call(model, 500.0, 0.001, success=True, quality_score=None)
    assert profiles[model].quality_score == initial


# ─────────────────────────────────────────────────────────────────────────────
# Trust factor dynamics
# ─────────────────────────────────────────────────────────────────────────────

def test_trust_factor_degrades_on_failure(collector, profiles, model):
    initial = profiles[model].trust_factor  # 1.0
    collector.record_call(model, 0.0, 0.0, success=False)
    assert profiles[model].trust_factor < initial


def test_trust_factor_recovers_on_success(collector, profiles, model):
    # Degrade first
    for _ in range(5):
        collector.record_call(model, 0.0, 0.0, success=False)
    degraded = profiles[model].trust_factor
    collector.record_call(model, 500.0, 0.001, success=True)
    assert profiles[model].trust_factor > degraded


def test_trust_factor_capped_at_1(collector, profiles, model):
    for _ in range(20):
        collector.record_call(model, 500.0, 0.001, success=True)
    assert profiles[model].trust_factor <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# record_validator_failure
# ─────────────────────────────────────────────────────────────────────────────

def test_record_validator_failure_increments_count(collector, profiles, model):
    collector.record_validator_failure(model)
    assert profiles[model].validator_fail_count == 1
    collector.record_validator_failure(model)
    assert profiles[model].validator_fail_count == 2


def test_record_validator_failure_unknown_model_is_noop(collector):
    """Should not raise for unknown models."""
    collector.record_validator_failure("does-not-exist")


# ─────────────────────────────────────────────────────────────────────────────
# record_policy_violation
# ─────────────────────────────────────────────────────────────────────────────

def test_record_policy_violation_degrades_trust(collector, profiles, model):
    initial = profiles[model].trust_factor
    collector.record_policy_violation(model)
    assert profiles[model].trust_factor < initial


def test_record_policy_violation_unknown_model_is_noop(collector):
    collector.record_policy_violation("unknown-model")


# ─────────────────────────────────────────────────────────────────────────────
# error_rate
# ─────────────────────────────────────────────────────────────────────────────

def test_error_rate_zero_when_no_calls(collector, model):
    assert collector.error_rate(model) == 0.0


def test_error_rate_zero_when_all_succeed(collector, model):
    collector.record_call(model, 500.0, 0.001, success=True)
    assert collector.error_rate(model) == 0.0


def test_error_rate_correct_after_failures(collector, model):
    collector.record_call(model, 500.0, 0.001, success=True)
    collector.record_call(model, 0.0, 0.0, success=False)
    assert collector.error_rate(model) == pytest.approx(0.5)


def test_error_rate_unknown_model_returns_zero(collector):
    assert collector.error_rate("nonexistent") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# get_profiles
# ─────────────────────────────────────────────────────────────────────────────

def test_get_profiles_returns_live_reference(collector, profiles, model):
    returned = collector.get_profiles()
    collector.record_call(model, 500.0, 0.001, success=True)
    # The returned dict is the same object — mutation is visible
    assert returned[model].call_count == 1
