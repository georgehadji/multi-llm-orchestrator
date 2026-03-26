"""Tests for per-model rate limiting."""

import time
from unittest.mock import patch

import pytest

from nadirclaw.rate_limit import ModelRateLimiter


class TestModelRateLimiter:
    """Unit tests for the ModelRateLimiter class."""

    def setup_method(self):
        self.limiter = ModelRateLimiter()
        # Clear any env-based config
        self.limiter._limits = {}
        self.limiter._default_rpm = 0
        self.limiter._hits.clear()

    def test_no_limit_allows_all(self):
        """With no limits configured, all requests pass."""
        for _ in range(200):
            assert self.limiter.check("gpt-4.1") is None

    def test_explicit_limit_enforced(self):
        """Requests beyond the configured RPM are blocked."""
        self.limiter.set_limit("gpt-4.1", 5)

        # First 5 should pass
        for i in range(5):
            result = self.limiter.check("gpt-4.1")
            assert result is None, f"Request {i+1} should pass"

        # 6th should be blocked
        retry_after = self.limiter.check("gpt-4.1")
        assert retry_after is not None
        assert retry_after > 0

    def test_default_rpm_applies_to_unconfigured_models(self):
        """The default RPM applies to models without explicit limits."""
        self.limiter.set_default(3)

        for _ in range(3):
            assert self.limiter.check("some-model") is None

        retry_after = self.limiter.check("some-model")
        assert retry_after is not None

    def test_explicit_limit_overrides_default(self):
        """Explicit per-model limit takes precedence over default."""
        self.limiter.set_default(2)
        self.limiter.set_limit("fast-model", 10)

        # fast-model should allow 10
        for _ in range(10):
            assert self.limiter.check("fast-model") is None
        assert self.limiter.check("fast-model") is not None

        # other-model uses default of 2
        for _ in range(2):
            assert self.limiter.check("other-model") is None
        assert self.limiter.check("other-model") is not None

    def test_independent_model_counters(self):
        """Each model has its own counter."""
        self.limiter.set_limit("model-a", 3)
        self.limiter.set_limit("model-b", 3)

        for _ in range(3):
            assert self.limiter.check("model-a") is None

        # model-a is exhausted
        assert self.limiter.check("model-a") is not None

        # model-b should still work
        for _ in range(3):
            assert self.limiter.check("model-b") is None

    def test_sliding_window_expires(self):
        """Hits expire after the 60-second window."""
        self.limiter.set_limit("test-model", 2)

        assert self.limiter.check("test-model") is None
        assert self.limiter.check("test-model") is None
        assert self.limiter.check("test-model") is not None

        # Simulate time passing (manually age the timestamps)
        with self.limiter._lock:
            q = self.limiter._hits["test-model"]
            # Move all timestamps back 61 seconds
            old_q = self.limiter._hits["test-model"]
            self.limiter._hits["test-model"] = type(old_q)(
                t - 61 for t in old_q
            )

        # Now requests should pass again
        assert self.limiter.check("test-model") is None

    def test_get_status(self):
        """Status endpoint returns correct info."""
        self.limiter.set_limit("gpt-4.1", 60)
        self.limiter.set_default(30)

        # Make a few requests
        self.limiter.check("gpt-4.1")
        self.limiter.check("gpt-4.1")
        self.limiter.check("unknown-model")

        status = self.limiter.get_status()
        assert status["default_rpm"] == 30
        assert "gpt-4.1" in status["models"]
        assert status["models"]["gpt-4.1"]["rpm_limit"] == 60
        assert status["models"]["gpt-4.1"]["current_rpm"] == 2
        assert status["models"]["gpt-4.1"]["remaining"] == 58

    def test_reset_single_model(self):
        """Reset clears counters for a specific model."""
        self.limiter.set_limit("model-a", 2)
        self.limiter.set_limit("model-b", 2)

        self.limiter.check("model-a")
        self.limiter.check("model-b")

        self.limiter.reset("model-a")

        status = self.limiter.get_status()
        assert status["models"]["model-a"]["current_rpm"] == 0
        assert status["models"]["model-b"]["current_rpm"] == 1

    def test_reset_all(self):
        """Reset without model clears all counters."""
        self.limiter.set_limit("model-a", 2)

        self.limiter.check("model-a")
        self.limiter.reset()

        status = self.limiter.get_status()
        assert status["models"]["model-a"]["current_rpm"] == 0

    def test_env_config_parsing(self):
        """Config is parsed correctly from env vars."""
        with patch.dict("os.environ", {
            "NADIRCLAW_MODEL_RATE_LIMITS": "gemini-3-flash-preview=30, gpt-4.1=60, ollama/llama3=120",
            "NADIRCLAW_DEFAULT_MODEL_RPM": "45",
        }):
            limiter = ModelRateLimiter()
            assert limiter.get_limit("gemini-3-flash-preview") == 30
            assert limiter.get_limit("gpt-4.1") == 60
            assert limiter.get_limit("ollama/llama3") == 120
            assert limiter.get_limit("unknown-model") == 45

    def test_env_config_invalid_entries_skipped(self):
        """Invalid entries in the config are skipped gracefully."""
        with patch.dict("os.environ", {
            "NADIRCLAW_MODEL_RATE_LIMITS": "good-model=50, bad-entry, no-number=abc, =10",
            "NADIRCLAW_DEFAULT_MODEL_RPM": "not-a-number",
        }):
            limiter = ModelRateLimiter()
            assert limiter.get_limit("good-model") == 50
            assert limiter.get_limit("bad-entry") == 0  # default 0 (invalid DEFAULT_MODEL_RPM)
            assert limiter._default_rpm == 0

    def test_get_limit_returns_zero_for_unlimited(self):
        """get_limit returns 0 for models with no limit."""
        assert self.limiter.get_limit("any-model") == 0

    def test_retry_after_is_positive(self):
        """retry_after is always at least 1 second."""
        self.limiter.set_limit("test", 1)
        self.limiter.check("test")
        retry = self.limiter.check("test")
        assert retry is not None
        assert retry >= 1
