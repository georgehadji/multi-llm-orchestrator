"""
Unified Resilience Policy Protocol
====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 1 — Resilience Unification (ARCHITECTURE_REMEDIATION_PLAN_V2).

Provides the abstract Protocol and concrete config dataclass for the
canonical, unified resilience policy that subsumes:
  - operations/resilience.py (existing tenacity-based retry)
  - circuit_breaker.py (3-state circuit breaker)
  - application/fallback_handler.py (binary health tracking)
  - operations/remediation.py (ordered remediation plans)
  - engine_core/escalation.py (quality-based escalation)
  - engine_core/sagas.py (inline retry loops)
  - cost_optimization/streaming_validator.py (hardcoded fallbacks)

Usage:
    from orchestrator.domain.resilience_policy import (
        ResiliencePolicyConfig,
        ResiliencePolicyPort,
        CircuitState,
        FallbackStrategy,
    )

    config = ResiliencePolicyConfig.for_task_type(TaskType.CODE_GEN)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, TypeVar

if TYPE_CHECKING:
    from ..models import Model, TaskType

T = TypeVar("T", covariant=False)

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    """Circuit breaker state machine states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FallbackStrategy(str, Enum):
    """Ordered strategies for the unified fallback/remediation pipeline."""

    RETRY_SAME_MODEL = "retry_same_model"
    FALLBACK_MODEL = "fallback_model"
    ESCALATE_QUALITY = "escalate_quality"
    DEGRADE_THRESHOLD = "degrade_threshold"
    ABORT = "abort"


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResiliencePolicyConfig:
    """
    Immutable configuration for a single resilience policy.

    Combines retry parameters, circuit-breaker thresholds, fallback-chain
    resolution rules, and remediation strategy ordering into one value object.

    Attributes:
        retries:                 Max retry attempts per model before falling back.
        timeout:                 Per-request timeout in seconds.
        backoff_base:            Exponential backoff multiplier.
        backoff_max:             Cap on backoff wait time (seconds).
        jitter:                  Add ±20 % random jitter to wait times.
        cb_failure_threshold:    Consecutive failures before circuit opens.
        cb_reset_timeout:        Seconds in OPEN before allowing a HALF_OPEN probe.
        cb_success_threshold:    Consecutive successes in HALF_OPEN to close again.
        fallback_strategies:     Ordered list of strategies for the remediation pipeline.
        retryable_exception_types:
                                 Tuple of exception types that trigger a retry.
    """

    retries: int = 2
    timeout: float = 60.0
    backoff_base: float = 2.0
    backoff_max: float = 30.0
    jitter: bool = True

    # Circuit breaker defaults (match circuit_breaker.py canonical values)
    cb_failure_threshold: int = 5
    cb_reset_timeout: float = 60.0
    cb_success_threshold: int = 2

    # Fallback / remediation strategy ordering
    fallback_strategies: tuple[FallbackStrategy, ...] = (
        FallbackStrategy.RETRY_SAME_MODEL,
        FallbackStrategy.FALLBACK_MODEL,
        FallbackStrategy.ESCALATE_QUALITY,
        FallbackStrategy.DEGRADE_THRESHOLD,
        FallbackStrategy.ABORT,
    )

    # Exceptions that trigger a retry (checked via ApplicationError.retriable flag at runtime)
    retryable_exception_types: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
    )

    # ── Presets ──────────────────────────────────────────────────────────────

    @classmethod
    def for_task_type(cls, task_type: TaskType) -> ResiliencePolicyConfig:
        """Return the preset config for a given TaskType."""
        # Deferred import to avoid circular dependency
        from ..models import TaskType as _TaskType

        _mapping: dict[_TaskType, ResiliencePolicyConfig] = {
            _TaskType.CODE_GEN: cls(retries=3, timeout=120.0, backoff_base=2.0, backoff_max=30.0),
            _TaskType.CODE_REVIEW: cls(retries=2, timeout=60.0, backoff_base=2.0, backoff_max=20.0),
            _TaskType.EVALUATE: cls(retries=2, timeout=45.0, backoff_base=1.5, backoff_max=10.0),
            _TaskType.REASONING: cls(retries=2, timeout=300.0, backoff_base=2.0, backoff_max=60.0),
        }
        return _mapping.get(task_type, cls())

    @classmethod
    def default(cls) -> ResiliencePolicyConfig:
        """Return the default policy."""
        return cls()

    def with_fallback_chain(self, chain: tuple[FallbackStrategy, ...]) -> ResiliencePolicyConfig:
        """Return a new config with the given fallback strategy ordering."""
        return ResiliencePolicyConfig(
            retries=self.retries,
            timeout=self.timeout,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            jitter=self.jitter,
            cb_failure_threshold=self.cb_failure_threshold,
            cb_reset_timeout=self.cb_reset_timeout,
            cb_success_threshold=self.cb_success_threshold,
            fallback_strategies=chain,
            retryable_exception_types=self.retryable_exception_types,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging / telemetry."""
        return {
            "retries": self.retries,
            "timeout_s": self.timeout,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "jitter": self.jitter,
            "cb_failure_threshold": self.cb_failure_threshold,
            "cb_reset_timeout_s": self.cb_reset_timeout,
            "cb_success_threshold": self.cb_success_threshold,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────


class UnifiedResiliencePolicy(Protocol):
    """
    Protocol that any unified resilience implementation must satisfy.

    The implementation (in operations/resilience.py) combines:
    - Retry with exponential backoff
    - Per-model circuit breaker gating
    - Fallback chain resolution via FALLBACK_CHAIN
    - Quality-based escalation
    - Remediation strategy ordering
    """

    async def execute_with_resilience(
        self,
        primary_callable: Callable[[], Awaitable[T]],
        fallback_callables: list[Callable[[], Awaitable[T]]] | None = None,
        *,
        model_id: str | None = None,
        config: ResiliencePolicyConfig | None = None,
    ) -> T:
        """
        Execute *primary_callable* with retry and circuit-breaker gating.

        If the primary exhausts its retries (or its circuit is OPEN), iterate
        through *fallback_callables* in order, applying the same per-callable
        retry policy.

        Args:
            primary_callable:   The primary async callable to execute.
            fallback_callables: Ordered fallback callables tried after
                                primary retries are exhausted.
            model_id:           Optional model identifier for circuit-breaker
                                lookups.  If None, circuit-breaker gating is
                                bypassed (backward-compatible path).
            config:             Override policy config.  If None, uses the
                                implementation's default.

        Returns:
            The result of the first successful callable.

        Raises:
            The last exception raised after all retries and fallbacks exhausted.
        """

    async def record_success(self, model_id: str) -> None:
        """Record a successful call for *model_id*, potentially closing its circuit."""

    async def record_failure(self, model_id: str, exception: Exception | None = None) -> None:
        """Record a failed call for *model_id*, potentially opening its circuit."""

    def is_model_available(self, model_id: str) -> bool:
        """Return True if *model_id*'s circuit is CLOSED or HALF_OPEN."""
        ...

    def tripped_models(self) -> list[str]:
        """Return all model IDs whose circuits are currently OPEN."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return aggregate resilience statistics."""
        ...
