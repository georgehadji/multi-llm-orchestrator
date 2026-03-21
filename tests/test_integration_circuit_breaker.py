"""
Tests for Integration Circuit Breaker
=====================================

Ensures circuit breaker correctly prevents silent failures
and provides clear error messages.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from orchestrator.integration_circuit_breaker import (
    IntegrationCircuitBreaker,
    IntegrationType,
    IntegrationState,
    IntegrationFailure,
    get_circuit_breaker,
)


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""
    
    @pytest.fixture
    def cb(self):
        """Create a circuit breaker for testing."""
        return IntegrationCircuitBreaker(
            strict_mode=False,
            fail_silently=False,
        )
    
    def test_initial_state_is_closed(self, cb):
        """Circuit starts in CLOSED state (normal operation)."""
        health = cb.get_health(IntegrationType.GITHUB)
        assert health.state == IntegrationState.CLOSED
        assert health.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_failure_count_increments(self, cb):
        """Each failure increments the counter."""
        call_count = 0
        
        @cb.guard(IntegrationType.GITHUB, "test_op")
        async def failing_op():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network failed")
        
        # Call 3 times (below threshold)
        for _ in range(3):
            await failing_op()
        
        health = cb.get_health(IntegrationType.GITHUB)
        assert health.consecutive_failures == 3
        assert call_count == 3  # All calls executed
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, cb):
        """Circuit opens after FAILURE_THRESHOLD failures."""
        cb.FAILURE_THRESHOLD = 2  # Lower for testing
        call_count = 0
        
        @cb.guard(IntegrationType.GITHUB, "test_op")
        async def failing_op():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network failed")
        
        # First 2 calls should execute
        await failing_op()
        await failing_op()
        
        health = cb.get_health(IntegrationType.GITHUB)
        assert health.state == IntegrationState.OPEN
        
        # Third call should NOT execute (circuit open)
        result = await failing_op()
        assert result is None
        assert call_count == 2  # No new calls


class TestStrictMode:
    """Test strict mode behavior."""
    
    @pytest.fixture
    def strict_cb(self):
        """Create a strict circuit breaker."""
        return IntegrationCircuitBreaker(strict_mode=True)
    
    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_failure(self, strict_cb):
        """Strict mode raises IntegrationFailure immediately."""
        
        @strict_cb.guard(
            IntegrationType.GITHUB,
            "create_check_run",
            remediation="Check GITHUB_TOKEN is valid"
        )
        async def failing_op():
            raise ConnectionError("Network failed")
        
        with pytest.raises(IntegrationFailure) as exc_info:
            await failing_op()
        
        assert "GITHUB_TOKEN is valid" in str(exc_info.value)
        assert exc_info.value.integration == IntegrationType.GITHUB
        assert exc_info.value.operation == "create_check_run"


class TestFailSilently:
    """Test fail-silently mode (legacy behavior)."""
    
    @pytest.fixture
    def silent_cb(self):
        """Create a silent circuit breaker."""
        return IntegrationCircuitBreaker(fail_silently=True)
    
    @pytest.mark.asyncio
    async def test_silent_mode_returns_none(self, silent_cb):
        """Silent mode returns None on failure without raising."""
        
        @silent_cb.guard(IntegrationType.SLACK, "send_alert")
        async def failing_op():
            raise ConnectionError("Slack down")
        
        result = await failing_op()
        assert result is None


class TestSuccessResetsCircuit:
    """Test that success resets the failure counter."""
    
    @pytest.mark.asyncio
    async def test_success_resets_failures(self):
        """A successful call resets the failure counter."""
        cb = IntegrationCircuitBreaker()
        call_count = 0
        should_fail = True
        
        @cb.guard(IntegrationType.GITHUB, "test_op")
        async def conditional_op():
            nonlocal call_count, should_fail
            call_count += 1
            if should_fail:
                raise ConnectionError("Failed")
            return "success"
        
        # First failure
        await conditional_op()
        assert cb.get_health(IntegrationType.GITHUB).consecutive_failures == 1
        
        # Second call succeeds
        should_fail = False
        result = await conditional_op()
        
        assert result == "success"
        assert cb.get_health(IntegrationType.GITHUB).consecutive_failures == 0


class TestConfigDetection:
    """Test configuration detection."""
    
    @pytest.mark.parametrize("env_var,integration,expected", [
        ("GITHUB_TOKEN", IntegrationType.GITHUB, True),
        ("GIT_TOKEN", IntegrationType.GITHUB, True),  # Fallback
        ("ORCHESTRATOR_SLACK_WEBHOOK_URL", IntegrationType.SLACK, True),
        ("ISSUE_TRACKER_API_TOKEN", IntegrationType.JIRA, True),
        ("ISSUE_TRACKER_API_TOKEN", IntegrationType.LINEAR, True),
    ])
    def test_config_detection(self, env_var, integration, expected):
        """Correctly detect integration configuration."""
        cb = IntegrationCircuitBreaker()
        
        with patch.dict("os.environ", {env_var: "test_token"}, clear=True):
            health = cb.get_health(integration)
            assert health.config_present == expected
    
    def test_no_config_detected(self):
        """Detect when integration is not configured."""
        cb = IntegrationCircuitBreaker()
        
        with patch.dict("os.environ", {}, clear=True):
            health = cb.get_health(IntegrationType.GITHUB)
            assert not health.config_present


class TestCIEnvironment:
    """Test CI-specific behaviors."""
    
    def test_github_annotation_format(self):
        """GitHub Actions annotations are correctly formatted."""
        from orchestrator.integration_circuit_breaker import IntegrationHealthReporter
        
        health = MagicMock()
        health.integration_type = IntegrationType.GITHUB
        health.config_present = False
        
        annotation = IntegrationHealthReporter.github_annotation(health)
        assert "::warning" in annotation
        assert "GITHUB" in annotation
    
    def test_strict_mode_from_env(self):
        """Strict mode can be enabled via environment variable."""
        with patch.dict("os.environ", {"ORCHESTRATOR_STRICT_INTEGRATIONS": "true"}):
            cb = get_circuit_breaker()
            assert cb.strict_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
