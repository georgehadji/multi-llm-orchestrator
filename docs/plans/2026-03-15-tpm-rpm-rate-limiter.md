# TPM/RPM Rate Limiter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-tenant, per-model TPM/RPM rate limiting using a sliding 60-second window, wired into the engine's primary generation call path.

**Architecture:** New `orchestrator/rate_limiter.py` module with `RateLimiter` (sliding window counters, in-memory) and `RateLimitExceeded` exception. Engine initializes a `RateLimiter` instance and wraps its primary API call with `check()` + `record()`. No external dependencies.

**Tech Stack:** Pure Python (`collections.deque`, `time.monotonic`), no Redis required.

---

## Context

**Sliding window algorithm:**
- Per (tenant, model) key: maintain a `deque` of `(timestamp, tokens)` tuples
- Before each `check()` or `record()`: evict entries older than 60 seconds from front of deque
- TPM check: `sum(tokens for _, tokens in window) + new_tokens > tpm_limit`
- RPM check: `len(window) >= rpm_limit`
- `record()`: appends `(time.monotonic(), tokens)` and evicts stale entries

**Engine integration point:** `_execute_task()` calls `self.client.call()` at line ~1940 for primary generation. We wrap ONLY this call with `check()` + `record()`. Other internal calls (decomposition, critique) are not rate-limited for now — they don't need per-tenant protection.

**Key constraint:** `RateLimitExceeded` must be non-fatal to the engine — caught and treated like a circuit-breaker failure (model marked as temporarily unavailable for that tenant, task fails with an appropriate status). Do NOT let it crash the engine loop.

---

## Task 1: RateLimiter module

**Files:**
- Create: `orchestrator/rate_limiter.py`
- Create: `tests/test_rate_limiter.py`

### Step 1: Write the failing tests

```python
# tests/test_rate_limiter.py
"""Tests for TPM/RPM sliding window rate limiter."""
import time
import pytest
from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded


def test_no_limits_set_always_passes():
    """Without set_limits(), check() never raises."""
    rl = RateLimiter()
    rl.check("tenant1", "deepseek-chat", tokens=1_000_000)  # no limits → no raise


def test_tpm_limit_raises_when_exceeded():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=100, rpm=1000)
    rl.record("tenant1", "deepseek-chat", tokens=90)   # 90 used
    with pytest.raises(RateLimitExceeded) as exc_info:
        rl.check("tenant1", "deepseek-chat", tokens=20)  # 90+20=110 > 100
    assert exc_info.value.limit_type == "tpm"
    assert exc_info.value.retry_after > 0


def test_rpm_limit_raises_when_exceeded():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1_000_000, rpm=2)
    rl.record("tenant1", "deepseek-chat", tokens=1)
    rl.record("tenant1", "deepseek-chat", tokens=1)
    with pytest.raises(RateLimitExceeded) as exc_info:
        rl.check("tenant1", "deepseek-chat", tokens=1)  # 3rd request > rpm=2
    assert exc_info.value.limit_type == "rpm"


def test_check_passes_when_under_limit():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1000, rpm=10)
    rl.record("tenant1", "deepseek-chat", tokens=500)
    rl.check("tenant1", "deepseek-chat", tokens=499)  # 500+499=999 < 1000 → no raise


def test_sliding_window_evicts_old_entries():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=100, rpm=10)

    # Manually inject a stale entry (61 seconds ago)
    import time
    key = ("tenant1", "deepseek-chat")
    rl._windows[key].append((time.monotonic() - 61, 90))  # stale

    # Should not count stale entry; 10 tokens is fine
    rl.check("tenant1", "deepseek-chat", tokens=10)  # no raise


def test_get_usage_returns_current_window_stats():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1000, rpm=10)
    rl.record("tenant1", "deepseek-chat", tokens=300)
    rl.record("tenant1", "deepseek-chat", tokens=200)

    usage = rl.get_usage("tenant1", "deepseek-chat")
    assert usage["tokens_used"] == 500
    assert usage["requests"] == 2


def test_separate_tenants_have_independent_limits():
    rl = RateLimiter()
    rl.set_limits("tenant_a", "deepseek-chat", tpm=100, rpm=10)
    rl.set_limits("tenant_b", "deepseek-chat", tpm=100, rpm=10)
    rl.record("tenant_a", "deepseek-chat", tokens=90)
    # tenant_b unaffected — should not raise
    rl.check("tenant_b", "deepseek-chat", tokens=90)
```

### Step 2: Run to confirm FAIL (ImportError)

Run: `python -m pytest tests/test_rate_limiter.py -v --no-cov`

Expected: `ModuleNotFoundError: cannot import name 'RateLimiter'`

### Step 3: Create `orchestrator/rate_limiter.py`

```python
"""
RateLimiter — Sliding-window TPM/RPM rate limiting per tenant and model.
=========================================================================
Implements in-memory, per-(tenant, model) rate limiting using a 60-second
sliding window. No external dependencies required.

Usage:
    limiter = RateLimiter()
    limiter.set_limits("acme", "deepseek-chat", tpm=50_000, rpm=100)

    # Before each API call:
    limiter.check("acme", "deepseek-chat", tokens=estimated_tokens)

    # After successful call:
    limiter.record("acme", "deepseek-chat", tokens=actual_tokens_used)
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from .log_config import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS: float = 60.0


class RateLimitExceeded(Exception):
    """Raised when a tenant/model would exceed its TPM or RPM limit."""

    def __init__(self, limit_type: str, retry_after: float, tenant: str, model: str) -> None:
        self.limit_type = limit_type      # "tpm" or "rpm"
        self.retry_after = retry_after    # seconds until oldest entry expires
        self.tenant = tenant
        self.model = model
        super().__init__(
            f"Rate limit exceeded ({limit_type}) for tenant={tenant!r} model={model!r}. "
            f"Retry after {retry_after:.1f}s"
        )


# (timestamp, tokens)
_Entry = Tuple[float, int]


class RateLimiter:
    """
    Per-(tenant, model) sliding-window rate limiter.

    Thread-safety: NOT thread-safe. For async use, callers must ensure
    serialised access (e.g. via asyncio.Lock) if needed. The engine's
    single-threaded async loop is sufficient in practice.
    """

    def __init__(self) -> None:
        # (tenant, model) → deque of (monotonic_timestamp, tokens)
        self._windows: Dict[Tuple[str, str], deque] = defaultdict(deque)
        # (tenant, model) → {"tpm": int, "rpm": int}
        self._limits: Dict[Tuple[str, str], Dict[str, int]] = {}

    # ── Configuration ─────────────────────────────────────────────────────

    def set_limits(self, tenant: str, model: str, tpm: int, rpm: int) -> None:
        """Set TPM and RPM limits for a (tenant, model) pair."""
        if tpm <= 0 or rpm <= 0:
            raise ValueError("tpm and rpm must be positive integers")
        self._limits[(tenant, model)] = {"tpm": tpm, "rpm": rpm}
        logger.debug("Rate limits set: tenant=%r model=%r tpm=%d rpm=%d", tenant, model, tpm, rpm)

    # ── Core API ──────────────────────────────────────────────────────────

    def check(self, tenant: str, model: str, tokens: int) -> None:
        """
        Check whether a request of `tokens` tokens is within limits.

        Raises RateLimitExceeded if the request would exceed TPM or RPM.
        No-ops if no limits have been set for this (tenant, model) pair.
        """
        limits = self._limits.get((tenant, model))
        if limits is None:
            return  # no limits configured → always allow

        window = self._windows[(tenant, model)]
        self._evict(window)

        # RPM check
        if len(window) >= limits["rpm"]:
            retry_after = self._retry_after(window)
            raise RateLimitExceeded("rpm", retry_after, tenant, model)

        # TPM check
        tokens_used = sum(t for _, t in window)
        if tokens_used + tokens > limits["tpm"]:
            retry_after = self._retry_after(window)
            raise RateLimitExceeded("tpm", retry_after, tenant, model)

    def record(self, tenant: str, model: str, tokens: int) -> None:
        """Record a completed request (called after a successful API call)."""
        window = self._windows[(tenant, model)]
        self._evict(window)
        window.append((time.monotonic(), tokens))

    def get_usage(self, tenant: str, model: str) -> Dict[str, int]:
        """Return current sliding-window usage for (tenant, model)."""
        window = self._windows[(tenant, model)]
        self._evict(window)
        return {
            "tokens_used": sum(t for _, t in window),
            "requests": len(window),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _evict(window: deque) -> None:
        """Remove entries older than the sliding window."""
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while window and window[0][0] < cutoff:
            window.popleft()

    @staticmethod
    def _retry_after(window: deque) -> float:
        """Seconds until the oldest entry expires."""
        if not window:
            return 0.0
        oldest_ts = window[0][0]
        return max(0.0, oldest_ts + _WINDOW_SECONDS - time.monotonic())
```

### Step 4: Run tests to confirm PASS

Run: `python -m pytest tests/test_rate_limiter.py -v --no-cov`

Expected: `7 passed`

### Step 5: Commit

```bash
git add orchestrator/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: add RateLimiter with sliding-window TPM/RPM limiting per tenant+model"
```

---

## Task 2: Wire RateLimiter into engine

**Files:**
- Modify: `orchestrator/engine.py` (`__init__` and `_execute_task`)
- Modify: `orchestrator/__init__.py` (export)
- Create: `tests/test_rate_limiter_engine_integration.py`

### Step 1: Write the failing test

```python
# tests/test_rate_limiter_engine_integration.py
"""Verify engine respects rate limits via RateLimiter integration."""
import pytest
from unittest.mock import MagicMock
from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded
from orchestrator.engine import Orchestrator


def test_engine_has_rate_limiter():
    orch = Orchestrator.__new__(Orchestrator)
    orch._rate_limiter = RateLimiter()
    assert isinstance(orch._rate_limiter, RateLimiter)


def test_engine_exposes_configure_rate_limits():
    orch = Orchestrator.__new__(Orchestrator)
    orch._rate_limiter = RateLimiter()
    # Must not raise
    orch.configure_rate_limits("acme", "deepseek-chat", tpm=50_000, rpm=100)
    usage = orch._rate_limiter.get_usage("acme", "deepseek-chat")
    assert usage["requests"] == 0


def test_rate_limiter_check_raises_correctly():
    """Engine's rate_limiter raises when limit exceeded."""
    orch = Orchestrator.__new__(Orchestrator)
    rl = RateLimiter()
    rl.set_limits("t1", "deepseek-chat", tpm=10, rpm=100)
    rl.record("t1", "deepseek-chat", tokens=10)  # hit limit
    orch._rate_limiter = rl

    with pytest.raises(RateLimitExceeded):
        orch._rate_limiter.check("t1", "deepseek-chat", tokens=1)
```

### Step 2: Run to confirm FAIL

Run: `python -m pytest tests/test_rate_limiter_engine_integration.py -v --no-cov`

Expected: `FAILED — test_engine_exposes_configure_rate_limits` (method doesn't exist yet)

### Step 3: Add to `Orchestrator.__init__`

After the `_hybrid_pipeline` block (around line 286), add:

```python
from .rate_limiter import RateLimiter
self._rate_limiter = RateLimiter()
```

### Step 4: Add `configure_rate_limits()` method to `Orchestrator`

Add this method near `hybrid_search()` (around line 730):

```python
def configure_rate_limits(
    self,
    tenant: str,
    model: str,
    tpm: int,
    rpm: int,
) -> None:
    """
    Set TPM/RPM rate limits for a specific tenant and model.

    Args:
        tenant: Tenant identifier (e.g. team name, org ID).
        model: Model identifier string (e.g. "deepseek-chat").
        tpm: Maximum tokens per minute for this tenant+model.
        rpm: Maximum requests per minute for this tenant+model.
    """
    self._rate_limiter.set_limits(tenant, model, tpm, rpm)
```

### Step 5: Run tests to confirm PASS

Run: `python -m pytest tests/test_rate_limiter_engine_integration.py -v --no-cov`

Expected: `3 passed`

### Step 6: Export from `orchestrator/__init__.py`

Add after the rate-limiter-adjacent imports:
```python
from .rate_limiter import RateLimiter, RateLimitExceeded
```

### Step 7: Run full new test suite

Run: `python -m pytest tests/test_rate_limiter.py tests/test_rate_limiter_engine_integration.py tests/test_preflight_integration.py tests/test_query_expander.py tests/test_hybrid_search_pipeline.py -v --no-cov`

Expected: all 28 pass.

### Step 8: Commit

```bash
git add orchestrator/engine.py orchestrator/__init__.py tests/test_rate_limiter_engine_integration.py
git commit -m "feat: wire RateLimiter into engine with configure_rate_limits() API"
```

---

## Verification

```bash
python -m pytest tests/test_rate_limiter.py tests/test_rate_limiter_engine_integration.py -v --no-cov
```

Expected: **10 tests, all passing**
