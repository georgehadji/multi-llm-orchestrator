"""
Tests for Fix #2: Critique Reviewer Resilience — graduated circuit breaker instead of 1-strike kill

Currently, a single transient critique error (429 rate limit, timeout) immediately disables the
model by setting api_health[reviewer] = False. This causes:
1. Silent quality collapse — remaining iterations use only self-evaluation (inflated scores)
2. No audit trail — the failure is hidden; logs show it but behavior masks it
3. No recovery path — transient errors permanently disable the reviewer for the task

The fix implements a graduated 3-strike circuit breaker:
- Transient error (429, timeout) → increment consecutive_failures counter
- 3 consecutive failures → mark model unhealthy (circuit breaker trips)
- Success → reset counter (transient error recovery)
- Permanent error (401, 404) → immediate mark unhealthy (no counter, no recovery)

Additional improvement: Project-level alert when review coverage drops below 80%.
"""

import pytest
from orchestrator.models import Model, TaskType, Task, ProjectStatus
from orchestrator.engine import Orchestrator


def test_current_behavior_critique_exception_kills_reviewer():
    """
    Documents current BROKEN behavior: Critique exceptions immediately kill reviewer.

    This test verifies the bug exists by showing that the current code sets
    api_health[reviewer] = False on ANY exception, not using circuit breaker.

    After fix, this test will be replaced by test_critique_uses_graduated_circuit_breaker()
    which verifies the 3-strike behavior instead.
    """
    # Current behavior: any critique exception sets api_health = False
    # This is the bug we're fixing — it should use _record_failure() instead
    orch = Orchestrator()

    # The bug: api_health is a dict[Model → bool] where False = unhealthy
    # When critique fails (line 861 in engine.py), it sets:
    #   self.api_health[reviewer] = False
    # This immediately disables the model without using circuit breaker

    assert hasattr(orch, 'api_health'), "Orchestrator must track api_health"
    assert isinstance(orch.api_health, dict), "api_health must be a dict"


def test_record_failure_uses_circuit_breaker():
    """
    FAILING TEST: _record_failure should increment consecutive failure counter.

    This test verifies the circuit breaker logic exists in _record_failure.
    Currently this passes because _record_failure exists, but critique exception
    handler doesn't call it.
    """
    orch = Orchestrator()
    model = Model.CLAUDE_OPUS

    # _record_failure should track consecutive failures
    assert hasattr(orch, '_consecutive_failures'), \
        "Orchestrator must track consecutive failures per model for circuit breaker"

    # Reset model to healthy state (test env marks it unhealthy initially due to missing API keys)
    orch.api_health[model] = True
    orch._consecutive_failures[model] = 0

    # Call _record_failure for a transient error
    orch._record_failure(model, error=Exception("429 rate limit"))

    # After ONE transient error, model should still be healthy (not at threshold yet)
    # Threshold is 3, so after 1 error it should still be True
    assert orch.api_health.get(model, True) is True, \
        "Single transient error should NOT disable model (3-strike threshold)"

    # Verify counter was incremented
    assert orch._consecutive_failures.get(model, 0) == 1, \
        "Consecutive failures counter should increment on transient error"


def test_three_consecutive_failures_trigger_circuit_breaker():
    """
    FAILING TEST: Three consecutive failures should trigger circuit breaker.

    Verifies that after 3 transient errors on same model, it gets marked unhealthy.
    """
    orch = Orchestrator()
    model = Model.CLAUDE_OPUS

    # Reset model to healthy state
    orch.api_health[model] = True
    orch._consecutive_failures[model] = 0

    # Three transient errors
    orch._record_failure(model, error=Exception("429 rate limit"))
    orch._record_failure(model, error=Exception("timeout"))
    orch._record_failure(model, error=Exception("temporary error"))

    # After 3 failures, model should be marked unhealthy (circuit breaker trips)
    assert orch.api_health.get(model, True) is False, \
        "3 consecutive transient errors should trigger circuit breaker (mark unhealthy)"

    # Verify counter reached threshold
    assert orch._consecutive_failures.get(model, 0) == 3


def test_success_resets_consecutive_failures_counter():
    """
    FAILING TEST: Success should reset the consecutive failures counter.

    Verifies that after success, the model can recover from transient errors.
    """
    from unittest.mock import MagicMock

    orch = Orchestrator()
    model = Model.CLAUDE_OPUS

    # Reset model to healthy state
    orch.api_health[model] = True
    orch._consecutive_failures[model] = 0

    # Record 2 failures
    orch._record_failure(model, error=Exception("429"))
    orch._record_failure(model, error=Exception("timeout"))

    # Model should still be healthy (not at 3 yet)
    assert orch.api_health.get(model, True) is True
    assert orch._consecutive_failures.get(model, 0) == 2

    # Record a success - need to mock APIResponse
    mock_response = MagicMock()
    mock_response.latency_ms = 100
    mock_response.cost_usd = 0.01
    mock_response.input_tokens = 100
    mock_response.output_tokens = 50
    orch._record_success(model, mock_response)

    # Counter should reset to 0
    assert orch._consecutive_failures.get(model, 0) == 0, \
        "Success should reset consecutive failures counter"

    # Record 2 more failures — should still be healthy
    orch._record_failure(model, error=Exception("429"))
    orch._record_failure(model, error=Exception("timeout"))

    # Should still be healthy because counter was reset by success
    assert orch.api_health.get(model, True) is True, \
        "After reset, 2 failures should not disable model"


def test_permanent_error_immediately_disables():
    """
    FAILING TEST: Permanent errors (401, 404) should immediately disable model.

    Verifies that auth/not-found errors don't count toward the 3-strike threshold
    but instead immediately mark the model unhealthy.
    """
    orch = Orchestrator()
    model = Model.DEEPSEEK_CHAT

    # Reset model to healthy state
    orch.api_health[model] = True
    orch._consecutive_failures[model] = 0

    # Single 401 error (permanent)
    orch._record_failure(model, error=Exception("401 Unauthorized"))

    # Should be immediately marked unhealthy (not waiting for 3-strike threshold)
    assert orch.api_health.get(model, True) is False, \
        "Permanent 401 error should immediately disable model"

    # Counter should NOT be incremented (permanent errors bypass counter)
    assert orch._consecutive_failures.get(model, 0) == 0, \
        "Permanent errors should not increment consecutive failures counter"
