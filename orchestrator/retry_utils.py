"""
Resilient Retry Logic with Tenacity
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Provides robust retry logic with exponential backoff, jitter, and
exception-specific retry strategies for LLM API calls.

Usage:
    from orchestrator.retry_utils import retry_with_backoff, async_retry
    
    @async_retry(max_attempts=3, timeout=60)
    async def call_llm(model, prompt):
        return await api.call(model, prompt)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Optional, Type, TypeVar
from functools import wraps

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random,
    before_sleep_log,
    after_log,
    RetryCallState,
    AsyncRetrying,
)

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════
# EXCEPTION CLASSES FOR RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════

class RetryableError(Exception):
    """Base class for retryable errors"""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded - should retry with backoff"""
    pass


class TimeoutError(RetryableError):
    """Request timeout - should retry with longer timeout"""
    pass


class ServiceUnavailableError(RetryableError):
    """Service temporarily unavailable - should retry"""
    pass


class NonRetryableError(Exception):
    """Non-retryable error - fail immediately"""
    pass


class ValidationError(NonRetryableError):
    """Validation error - retry won't help"""
    pass


class AuthenticationError(NonRetryableError):
    """Authentication failed - retry won't help"""
    pass


# ═══════════════════════════════════════════════════════════════════
# DECORATORS FOR ASYNC RETRIES
# ═══════════════════════════════════════════════════════════════════

def async_retry(
    max_attempts: int = 3,
    timeout: int = 300,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[Type[Exception], ...] = (
        RetryableError,
        TimeoutError,
        RateLimitError,
        ServiceUnavailableError,
    ),
    logger_name: str = __name__
) -> Callable:
    """
    Decorator for async functions with exponential backoff retry.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        timeout: Maximum total timeout in seconds (default: 300)
        min_wait: Minimum wait time between retries in seconds (default: 1.0)
        max_wait: Maximum wait time between retries in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to wait time (default: True)
        retryable_exceptions: Tuple of exception types to retry on
        logger_name: Logger name for retry logging
    
    Returns:
        Decorated async function with retry logic
    
    Example:
        @async_retry(max_attempts=3, timeout=60)
        async def call_llm(model, prompt):
            return await api.call(model, prompt)
    """
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create retry strategy
            wait_strategy = wait_exponential(
                multiplier=exponential_base,
                min=min_wait,
                max=max_wait
            )
            
            if jitter:
                # Add random jitter (±20% of wait time)
                wait_strategy = wait_strategy + wait_random(
                    min=min_wait * 0.2,
                    max=max_wait * 0.2
                )
            
            # Create retry instance
            retrying = AsyncRetrying(
                stop=stop_after_attempt(max_attempts) | stop_after_delay(timeout),
                wait=wait_strategy,
                retry=retry_if_exception_type(retryable_exceptions),
                before_sleep=before_sleep_log(
                    logging.getLogger(logger_name),
                    logging.WARNING
                ),
                after=after_log(
                    logging.getLogger(logger_name),
                    logging.INFO
                )
            )
            
            # Execute with retries
            async for attempt in retrying:
                with attempt:
                    result = await func(*args, **kwargs)
                    return result
            
            # Should not reach here, but just in case
            raise RuntimeError(f"Failed after {max_attempts} attempts")
        
        return wrapper
    
    return decorator


def retry_with_backoff(
    func: Callable[..., Any],
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[Type[Exception], ...] = (RetryableError,)
) -> Callable[..., Any]:
    """
    Wrap a function with retry logic (for sync functions).

    .. warning::
        This wrapper uses ``time.sleep()`` for backoff and MUST NOT be called
        from within an ``async def`` context — doing so blocks the event loop.
        Use :func:`retry_async` (tenacity-based) for async callers instead.

    Args:
        func: Function to wrap
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait time
        max_wait: Maximum wait time
        exponential_base: Exponential backoff base
        jitter: Add jitter
        retryable_exceptions: Exceptions to retry on
    
    Returns:
        Wrapped function with retry logic
    """
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wait_time = min_wait
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                if attempt == max_attempts:
                    logger.error(
                        f"Failed after {max_attempts} attempts: {e}",
                        exc_info=True
                    )
                    raise
                
                # Calculate wait time with optional jitter
                actual_wait = wait_time
                if jitter:
                    actual_wait += random.uniform(
                        -wait_time * 0.2,
                        wait_time * 0.2
                    )
                
                logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {actual_wait:.2f}s...",
                    exc_info=True
                )
                
                # time.sleep() is intentional — this is a sync-only wrapper.
                # See docstring warning: must not be called from async context.
                time.sleep(actual_wait)
                wait_time = min(wait_time * exponential_base, max_wait)
        
        # Should not reach here
        raise RuntimeError(f"Failed after {max_attempts} attempts")
    
    return wrapper


# ═══════════════════════════════════════════════════════════════════
# SPECIALIZED RETRY DECORATORS FOR COMMON USE CASES
# ═══════════════════════════════════════════════════════════════════

# LLM API Call Retry (for timeouts and rate limits)
llm_retry = async_retry(
    max_attempts=3,
    timeout=120,
    min_wait=2.0,
    max_wait=30.0,
    retryable_exceptions=(TimeoutError, RateLimitError, ServiceUnavailableError)
)


# Database Operation Retry (for locks and conflicts)
db_retry = async_retry(
    max_attempts=5,
    timeout=30,
    min_wait=0.5,
    max_wait=5.0,
    retryable_exceptions=(RetryableError,)  # Add sqlite3.OperationalError for DB locks
)


# Cache Operation Retry (for cache misses and locks)
cache_retry = async_retry(
    max_attempts=3,
    timeout=15,
    min_wait=0.2,
    max_wait=2.0,
    retryable_exceptions=(RetryableError,)
)


# ═══════════════════════════════════════════════════════════════════
# CUSTOM RETRY STRATEGIES
# ═══════════════════════════════════════════════════════════════════

class RetryStrategy:
    """Configurable retry strategy builder"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        timeout: int = 300,
        min_wait: float = 1.0,
        max_wait: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def create_decorator(
        self,
        retryable_exceptions: tuple[Type[Exception], ...] = (RetryableError,)
    ) -> Callable:
        """Create retry decorator with this strategy"""
        return async_retry(
            max_attempts=self.max_attempts,
            timeout=self.timeout,
            min_wait=self.min_wait,
            max_wait=self.max_wait,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            retryable_exceptions=retryable_exceptions
        )
    
    @classmethod
    def aggressive(cls) -> 'RetryStrategy':
        """Aggressive retry strategy (more attempts, longer waits)"""
        return cls(
            max_attempts=5,
            timeout=300,
            min_wait=2.0,
            max_wait=60.0,
            exponential_base=2.0
        )
    
    @classmethod
    def conservative(cls) -> 'RetryStrategy':
        """Conservative retry strategy (fewer attempts, shorter waits)"""
        return cls(
            max_attempts=2,
            timeout=60,
            min_wait=0.5,
            max_wait=10.0,
            exponential_base=1.5
        )
    
    @classmethod
    def fast_fail(cls) -> 'RetryStrategy':
        """Fast-fail strategy (minimal retries for latency-critical ops)"""
        return cls(
            max_attempts=1,
            timeout=30,
            min_wait=0.1,
            max_wait=1.0,
            jitter=False
        )


# ═══════════════════════════════════════════════════════════════════
# CALLBACKS FOR RETRY LOGGING AND MONITORING
# ═══════════════════════════════════════════════════════════════════

def on_retry_attempt(retry_state: RetryCallState) -> None:
    """
    Callback executed on each retry attempt.
    
    Use this for:
    - Metrics collection
    - Alerting on repeated failures
    - Circuit breaker integration
    """
    if retry_state.attempt_number > 3:
        # Alert on excessive retries
        logger.warning(
            f"Excessive retries detected: {retry_state.fn.__name__} "
            f"attempted {retry_state.attempt_number} times"
        )
    
    # Log retry details
    logger.debug(
        f"Retry attempt {retry_state.attempt_number} for {retry_state.fn.__name__}",
        extra={
            'function': retry_state.fn.__name__,
            'attempt': retry_state.attempt_number,
            'max_attempts': retry_state.stop.max_attempts,
            'exception': str(retry_state.outcome.exception()) if retry_state.outcome else None
        }
    )


def on_retry_success(retry_state: RetryCallState) -> None:
    """
    Callback executed when retry succeeds.
    
    Use this for:
    - Success metrics
    - Recovery logging
    """
    if retry_state.attempt_number > 1:
        logger.info(
            f"Operation succeeded after {retry_state.attempt_number} attempts: "
            f"{retry_state.fn.__name__}"
        )


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def is_retryable_exception(exc: Exception) -> bool:
    """Check if exception is retryable"""
    return isinstance(exc, RetryableError)


def should_retry(
    exc: Exception,
    attempt: int,
    max_attempts: int,
    elapsed_time: float,
    timeout: float
) -> bool:
    """
    Determine if operation should be retried.
    
    Args:
        exc: Exception that occurred
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        elapsed_time: Time elapsed since first attempt
        timeout: Maximum total timeout
    
    Returns:
        True if should retry, False otherwise
    """
    # Check if exception is retryable
    if not is_retryable_exception(exc):
        return False
    
    # Check attempt limit
    if attempt >= max_attempts:
        return False
    
    # Check timeout
    if elapsed_time >= timeout:
        return False
    
    return True


def calculate_backoff(
    attempt: int,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """
    Calculate backoff time for retry.
    
    Args:
        attempt: Current attempt number
        min_wait: Minimum wait time
        max_wait: Maximum wait time
        exponential_base: Exponential base
        jitter: Add jitter
    
    Returns:
        Wait time in seconds
    """
    # Exponential backoff
    wait_time = min_wait * (exponential_base ** (attempt - 1))
    
    # Cap at maximum
    wait_time = min(wait_time, max_wait)
    
    # Add jitter
    if jitter:
        jitter_range = wait_time * 0.2
        wait_time += random.uniform(-jitter_range, jitter_range)
    
    return max(0.1, wait_time)  # Ensure minimum 0.1s


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: LLM API call with retry
    @llm_retry
    async def call_llm_api(model: str, prompt: str) -> str:
        """Example LLM API call with automatic retries"""
        # Simulate API call that may fail
        import random
        if random.random() < 0.7:  # 70% failure rate for demo
            raise TimeoutError("API timeout")
        return f"Response from {model}"
    
    # Example: Database operation with retry
    @db_retry
    async def query_database(query: str) -> list[dict]:
        """Example database query with automatic retries"""
        # Simulate database lock
        import random
        if random.random() < 0.5:
            raise RetryableError("Database locked")
        return [{"result": "data"}]
    
    # Example: Custom retry strategy
    custom_strategy = RetryStrategy.aggressive()
    
    @custom_strategy.create_decorator(
        retryable_exceptions=(TimeoutError, RateLimitError)
    )
    async def critical_operation(data: str) -> str:
        """Critical operation with aggressive retry strategy"""
        return f"Processed: {data}"
    
    # Run examples
    async def main():
        try:
            result = await call_llm_api("gpt-4", "Hello")
            print(f"LLM Result: {result}")
            
            result = await query_database("SELECT *")
            print(f"DB Result: {result}")
            
            result = await critical_operation("test")
            print(f"Critical Result: {result}")
        except Exception as e:
            print(f"Operation failed: {e}")
    
    asyncio.run(main())
