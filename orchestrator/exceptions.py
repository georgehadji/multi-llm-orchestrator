"""
Exception Hierarchy
===================
Explicit exception hierarchy for proper error handling and observability.

Architecture:
    ApplicationError (base)
    ├── ConfigurationError    # Invalid/missing configuration
    ├── OrchestratorError     # Core orchestration failures
    ├── ModelError            # LLM provider errors
    │   ├── ModelUnavailableError
    │   ├── RateLimitError
    │   └── TokenLimitError
    ├── TaskError             # Task execution errors
    │   ├── TaskValidationError
    │   ├── TaskTimeoutError
    │   └── TaskRetryExhaustedError
    ├── PolicyError           # Policy enforcement errors
    └── CacheError            # Cache/storage errors

Usage:
    try:
        result = await execute_task(task)
    except ModelUnavailableError as e:
        # Retry with different model
        logger.warning(f"Model {e.model} unavailable, trying fallback")
    except TaskTimeoutError as e:
        # Handle timeout specifically
        metrics.increment("task.timeout")
    except OrchestratorError as e:
        # Handle any orchestrator error
        logger.error(f"Orchestration failed: {e}")
"""
from __future__ import annotations

from typing import Any, Optional


class ApplicationError(Exception):
    """Base exception for all application errors.
    
    Provides structured error information for logging and debugging.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code
        details: Additional context (model name, task ID, etc.)
        retriable: Whether the operation can be retried
    """
    
    code: str = "UNKNOWN_ERROR"
    retriable: bool = False
    
    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        retriable: Optional[bool] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.details = details or {}
        self.retriable = retriable if retriable is not None else self.retriable
        self.cause = cause
        
        # Auto-populate common fields
        if cause:
            self.details["cause_type"] = type(cause).__name__
            self.details["cause_message"] = str(cause)
    
    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")
        return " ".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.code,
            "message": self.message,
            "retriable": self.retriable,
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Errors
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigurationError(ApplicationError):
    """Configuration is invalid, missing, or cannot be loaded."""
    code = "CONFIGURATION_ERROR"
    retriable = False


class MissingAPIKeyError(ConfigurationError):
    """Required API key is not configured."""
    code = "MISSING_API_KEY"
    
    def __init__(self, provider: str, **kwargs):
        super().__init__(
            f"API key for {provider} is not configured",
            details={"provider": provider, **kwargs.get("details", {})},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator Errors
# ═══════════════════════════════════════════════════════════════════════════════

class OrchestratorError(ApplicationError):
    """Core orchestration failure."""
    code = "ORCHESTRATOR_ERROR"
    retriable = True


class BudgetExceededError(OrchestratorError):
    """Budget limit has been exceeded."""
    code = "BUDGET_EXCEEDED"
    retriable = False
    
    def __init__(self, spent: float, limit: float, **kwargs):
        super().__init__(
            f"Budget exceeded: ${spent:.2f} / ${limit:.2f}",
            details={"spent_usd": spent, "limit_usd": limit, **kwargs.get("details", {})},
            **kwargs,
        )


class TimeoutError(OrchestratorError):
    """Operation timed out."""
    code = "TIMEOUT"
    retriable = True
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            details={"operation": operation, "timeout": timeout_seconds, **kwargs.get("details", {})},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Model Errors
# ═══════════════════════════════════════════════════════════════════════════════

class ModelError(ApplicationError):
    """LLM provider or model-related error."""
    code = "MODEL_ERROR"
    retriable = True


class ModelUnavailableError(ModelError):
    """Model is temporarily unavailable."""
    code = "MODEL_UNAVAILABLE"
    retriable = True
    
    def __init__(self, model: str, reason: Optional[str] = None, **kwargs):
        message = f"Model '{model}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            details={"model": model, "reason": reason, **kwargs.get("details", {})},
            **kwargs,
        )


class RateLimitError(ModelError):
    """Rate limit exceeded for provider."""
    code = "RATE_LIMIT_EXCEEDED"
    retriable = True
    
    def __init__(self, provider: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            f"Rate limit exceeded for {provider}",
            details={"provider": provider, "retry_after": retry_after, **kwargs.get("details", {})},
            **kwargs,
        )


class TokenLimitError(ModelError):
    """Token limit exceeded for model."""
    code = "TOKEN_LIMIT_EXCEEDED"
    retriable = False
    
    def __init__(self, model: str, tokens: int, limit: int, **kwargs):
        super().__init__(
            f"Token limit exceeded for {model}: {tokens} > {limit}",
            details={"model": model, "tokens": tokens, "limit": limit, **kwargs.get("details", {})},
            **kwargs,
        )


class AuthenticationError(ModelError):
    """API authentication failed."""
    code = "AUTHENTICATION_ERROR"
    retriable = False


# ═══════════════════════════════════════════════════════════════════════════════
# Task Errors
# ═══════════════════════════════════════════════════════════════════════════════

class TaskError(ApplicationError):
    """Task execution error."""
    code = "TASK_ERROR"
    retriable = True


class TaskValidationError(TaskError):
    """Task validation failed."""
    code = "TASK_VALIDATION_ERROR"
    retriable = False


class TaskTimeoutError(TaskError):
    """Task execution timed out."""
    code = "TASK_TIMEOUT"
    retriable = True
    
    def __init__(self, task_id: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Task '{task_id}' timed out after {timeout_seconds}s",
            details={"task_id": task_id, "timeout": timeout_seconds, **kwargs.get("details", {})},
            **kwargs,
        )


class TaskRetryExhaustedError(TaskError):
    """All retry attempts exhausted."""
    code = "RETRY_EXHAUSTED"
    retriable = False
    
    def __init__(self, task_id: str, attempts: int, **kwargs):
        super().__init__(
            f"Task '{task_id}' failed after {attempts} attempts",
            details={"task_id": task_id, "attempts": attempts, **kwargs.get("details", {})},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Errors
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyError(ApplicationError):
    """Policy enforcement error."""
    code = "POLICY_ERROR"
    retriable = False


class PolicyViolationError(PolicyError):
    """Policy constraint was violated."""
    code = "POLICY_VIOLATION"
    
    def __init__(self, policy: str, constraint: str, **kwargs):
        super().__init__(
            f"Policy '{policy}' violated: {constraint}",
            details={"policy": policy, "constraint": constraint, **kwargs.get("details", {})},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Cache/Storage Errors
# ═══════════════════════════════════════════════════════════════════════════════

class CacheError(ApplicationError):
    """Cache or storage operation failed."""
    code = "CACHE_ERROR"
    retriable = True


class StateError(ApplicationError):
    """State persistence/loading failed."""
    code = "STATE_ERROR"
    retriable = True
