"""
Nexus Search — Circuit Breaker
===============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Circuit breaker pattern for search failures to prevent cascade failures.

Features:
- Automatic failure detection
- Configurable failure threshold
- Recovery timeout with half-open state
- Prevents cascade failures
- Metrics and monitoring

Usage:
    from orchestrator.nexus_search.optimization import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    results = await breaker.call(nexus_search, query)
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from orchestrator.nexus_search.models import SearchResults

logger = logging.getLogger("orchestrator.nexus_search")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Recovery testing, limited requests allowed


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for search operations.

    Prevents cascade failures by stopping requests when failure rate
    exceeds threshold. Automatically recovers after timeout period.

    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Failure threshold exceeded, requests blocked
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        results = await breaker.call(search_function, query)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # State tracking
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: float | None = None
        self.half_open_calls = 0

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._circuit_opens = 0

    async def call(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> SearchResults:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        self._total_calls += 1

        # Check if we should allow the call
        if not self._allow_call():
            self._total_failures += 1
            logger.warning(
                f"Circuit breaker OPEN - rejecting call. "
                f"Retry after {self._time_until_retry():.0f}s"
            )
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN. Retry after {self._time_until_retry():.0f}s"
            )

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Success - reset failure count
            self._on_success()

            return result

        except Exception as e:
            # Failure - increment failure count
            self._on_failure()

            # Re-raise the exception
            raise

    def _allow_call(self) -> bool:
        """
        Check if call should be allowed.

        Returns:
            True if call allowed, False otherwise
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._time_since_last_failure() >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def _on_success(self):
        """Handle successful call."""
        self.successes += 1
        self._total_successes += 1
        self.failures = 0

        if self.state == CircuitState.HALF_OPEN:
            # Successful call in half-open state - close circuit
            logger.info("Circuit breaker transitioning to CLOSED (recovery successful)")
            self.state = CircuitState.CLOSED
            self.half_open_calls = 0

        logger.debug(f"Circuit breaker success (state={self.state.value})")

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self._total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state - reopen circuit
            logger.warning("Circuit breaker transitioning to OPEN (recovery failed)")
            self.state = CircuitState.OPEN
            self._circuit_opens += 1
            self.half_open_calls = 0

        elif self.state == CircuitState.CLOSED:
            # Check if failure threshold exceeded
            if self.failures >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker transitioning to OPEN "
                    f"({self.failures} failures >= threshold {self.failure_threshold})"
                )
                self.state = CircuitState.OPEN
                self._circuit_opens += 1

    def _time_since_last_failure(self) -> float:
        """Get time since last failure in seconds."""
        if self.last_failure_time is None:
            return float("inf")
        return time.time() - self.last_failure_time

    def _time_until_retry(self) -> float:
        """Get time until retry is allowed."""
        if self.last_failure_time is None:
            return 0
        elapsed = self._time_since_last_failure()
        return max(0, self.recovery_timeout - elapsed)

    def reset(self):
        """Reset circuit breaker to initial state."""
        logger.info("Circuit breaker manually reset")
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failures": self.failures,
            "successes": self.successes,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "circuit_opens": self._circuit_opens,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_until_retry": self._time_until_retry() if self.state == CircuitState.OPEN else 0,
        }


# Global instances for different operations
_search_breaker: CircuitBreaker | None = None
_research_breaker: CircuitBreaker | None = None


def get_search_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 60,
) -> CircuitBreaker:
    """
    Get circuit breaker for search operations.

    Args:
        failure_threshold: Failures before opening
        recovery_timeout: Seconds until recovery attempt

    Returns:
        CircuitBreaker instance
    """
    global _search_breaker
    if _search_breaker is None:
        _search_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _search_breaker


def get_research_breaker(
    failure_threshold: int = 5,  # More tolerant for research
    recovery_timeout: int = 120,  # Longer recovery for research
) -> CircuitBreaker:
    """
    Get circuit breaker for research operations.

    Args:
        failure_threshold: Failures before opening
        recovery_timeout: Seconds until recovery attempt

    Returns:
        CircuitBreaker instance
    """
    global _research_breaker
    if _research_breaker is None:
        _research_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _research_breaker


async def search_with_breaker(
    search_func: Callable,
    query: str,
    **kwargs: Any,
) -> SearchResults:
    """
    Execute search with circuit breaker protection.

    Convenience function for common search pattern.

    Args:
        search_func: Search function to call
        query: Search query
        **kwargs: Additional arguments for search function

    Returns:
        SearchResults

    Raises:
        CircuitBreakerError: If circuit is open
    """
    breaker = get_search_breaker()
    return await breaker.call(search_func, query, **kwargs)


async def research_with_breaker(
    research_func: Callable,
    query: str,
    **kwargs: Any,
) -> Any:
    """
    Execute research with circuit breaker protection.

    Convenience function for common research pattern.

    Args:
        research_func: Research function to call
        query: Research query
        **kwargs: Additional arguments for research function

    Returns:
        Research results
    """
    breaker = get_research_breaker()
    return await breaker.call(research_func, query, **kwargs)
