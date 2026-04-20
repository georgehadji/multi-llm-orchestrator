"""
Circuit Breaker — async, per-key fail-fast for external services.
=================================================================
Prevents thundering-herd retries when a provider is down.

States:
  CLOSED  → normal; calls pass through
  OPEN    → tripped; calls raise CircuitBreakerOpen immediately (fail-fast)
  HALF_OPEN → one probe call allowed to test recovery

Usage:
    cb = CircuitBreaker(name="openrouter", failure_threshold=5, reset_timeout=60.0)

    async with cb:          # raises CircuitBreakerOpen when OPEN
        response = await client.call(...)

    # Or manually:
    async with cb.context("openai/gpt-4o"):
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from .exceptions import ApplicationError

logger = logging.getLogger("orchestrator.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(ApplicationError):
    """Raised when a circuit breaker is OPEN (fail-fast)."""

    code = "CIRCUIT_BREAKER_OPEN"
    retriable = False

    def __init__(self, name: str, reset_in: float, **kwargs):
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — retry in {reset_in:.1f}s",
            details={"circuit": name, "reset_in_seconds": reset_in},
            **kwargs,
        )


@dataclass
class _BreakerState:
    failures: int = 0
    successes: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at: float = 0.0
    last_failure_at: float = 0.0
    probe_in_flight: bool = False


class CircuitBreaker:
    """
    Per-key async circuit breaker.

    Args:
        name:              Identifier used in logs and exceptions.
        failure_threshold: Consecutive failures before tripping (default 5).
        reset_timeout:     Seconds in OPEN before allowing one probe (default 60.0).
        success_threshold: Consecutive successes in HALF_OPEN to close again (default 2).
        half_open_timeout: Max seconds a HALF_OPEN probe may take before re-tripping (default 30.0).
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        half_open_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_timeout = half_open_timeout

        self._lock = asyncio.Lock()
        self._state = _BreakerState()

        # Metrics (monotonic counters — never decremented)
        self.total_calls: int = 0
        self.total_failures: int = 0
        self.total_successes: int = 0
        self.trip_count: int = 0

    # ─────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._state.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state.state == CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        return self._state.state

    def time_until_reset(self) -> float:
        """Seconds remaining until OPEN transitions to HALF_OPEN. 0 if not OPEN."""
        if self._state.state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._state.opened_at
        remaining = self.reset_timeout - elapsed
        return max(0.0, remaining)

    async def record_success(self) -> None:
        async with self._lock:
            self.total_successes += 1
            self._state.successes += 1
            self._state.failures = 0
            if self._state.state == CircuitState.HALF_OPEN:
                if self._state.successes >= self.success_threshold:
                    self._close()
            elif self._state.state == CircuitState.OPEN:
                # Shouldn't happen; close anyway
                self._close()

    async def record_failure(self, exc: Exception | None = None) -> None:
        async with self._lock:
            self.total_failures += 1
            self._state.failures += 1
            self._state.successes = 0
            self._state.last_failure_at = time.monotonic()
            if self._state.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self._state.failures >= self.failure_threshold:
                    self._open(exc)

    async def check(self) -> None:
        """
        Raise CircuitBreakerOpen if the circuit is OPEN and not yet ready to probe.
        Transition OPEN → HALF_OPEN when reset_timeout has elapsed.
        """
        async with self._lock:
            state = self._state.state
            if state == CircuitState.CLOSED:
                self.total_calls += 1
                return
            if state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._state.opened_at
                if elapsed >= self.reset_timeout:
                    if not self._state.probe_in_flight:
                        self._state.state = CircuitState.HALF_OPEN
                        self._state.probe_in_flight = True
                        self._state.successes = 0
                        self.total_calls += 1
                        logger.info("Circuit breaker '%s' → HALF_OPEN (probe)", self.name)
                        return
                    # Another probe already in flight; reject this call
                raise CircuitBreakerOpen(self.name, reset_in=self.reset_timeout - elapsed)
            if state == CircuitState.HALF_OPEN:
                # Only one probe at a time; block additional callers
                if self._state.probe_in_flight:
                    raise CircuitBreakerOpen(self.name, reset_in=0.0)
                self.total_calls += 1

    @asynccontextmanager
    async def context(self) -> AsyncIterator[None]:
        """
        Async context manager that checks the circuit before the call and
        records success/failure based on the outcome.

        Usage:
            async with cb.context():
                result = await external_call()
        """
        await self.check()
        try:
            yield
            await self.record_success()
        except CircuitBreakerOpen:
            raise
        except Exception as exc:
            await self.record_failure(exc)
            raise

    def stats(self) -> dict[str, object]:
        """Return current metrics snapshot."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failures_consecutive": self._state.failures,
            "successes_consecutive": self._state.successes,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "trip_count": self.trip_count,
            "time_until_reset_s": self.time_until_reset(),
        }

    # ─────────────────────────────────────────────────────
    # Internal state transitions
    # ─────────────────────────────────────────────────────

    def _open(self, cause: Exception | None = None) -> None:
        self._state.state = CircuitState.OPEN
        self._state.opened_at = time.monotonic()
        self._state.probe_in_flight = False
        self.trip_count += 1
        cause_str = f": {cause}" if cause else ""
        logger.warning(
            "Circuit breaker '%s' → OPEN after %d consecutive failures%s "
            "(will probe in %.0fs)",
            self.name,
            self._state.failures,
            cause_str,
            self.reset_timeout,
        )

    def _close(self) -> None:
        self._state.state = CircuitState.CLOSED
        self._state.failures = 0
        self._state.successes = 0
        self._state.probe_in_flight = False
        logger.info("Circuit breaker '%s' → CLOSED (recovered)", self.name)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model registry
# ─────────────────────────────────────────────────────────────────────────────


class CircuitBreakerRegistry:
    """
    Thread-safe registry of per-model circuit breakers.

    Each LLM model (identified by its string value) gets its own breaker so
    that a single failing provider cannot trip the breaker for all models.

    Usage:
        registry = CircuitBreakerRegistry()

        async with registry.get("openai/gpt-4o").context():
            result = await call_openai(...)

        # Check which models are currently open:
        tripped = registry.tripped_models()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._success_threshold = success_threshold
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, model_id: str) -> CircuitBreaker:
        """Return the breaker for *model_id*, creating it on first access."""
        async with self._lock:
            if model_id not in self._breakers:
                self._breakers[model_id] = CircuitBreaker(
                    name=f"model:{model_id}",
                    failure_threshold=self._failure_threshold,
                    reset_timeout=self._reset_timeout,
                    success_threshold=self._success_threshold,
                )
            return self._breakers[model_id]

    def get_sync(self, model_id: str) -> CircuitBreaker:
        """
        Synchronous accessor — safe only when called before any concurrent async
        access (e.g. during engine __init__).  Prefer ``get()`` in async code.
        """
        if model_id not in self._breakers:
            self._breakers[model_id] = CircuitBreaker(
                name=f"model:{model_id}",
                failure_threshold=self._failure_threshold,
                reset_timeout=self._reset_timeout,
                success_threshold=self._success_threshold,
            )
        return self._breakers[model_id]

    def tripped_models(self) -> list[str]:
        """Return model IDs whose circuit breaker is currently OPEN."""
        return [
            mid
            for mid, cb in self._breakers.items()
            if cb._state.state != CircuitState.CLOSED
        ]

    def all_stats(self) -> dict[str, dict[str, object]]:
        """Return stats dict keyed by model_id."""
        return {mid: cb.stats() for mid, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Force-close all breakers (for testing / manual recovery)."""
        for cb in self._breakers.values():
            cb._close()
        logger.info("CircuitBreakerRegistry: all breakers reset")
