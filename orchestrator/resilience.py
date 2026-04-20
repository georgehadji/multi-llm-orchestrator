"""
Resilience Layer — Unified retry, fallback, and timeout policy
===============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Provides a single value object (ResiliencePolicy) that encapsulates all
retry/fallback/timeout behavior, plus a task-aware preset table (RetryTemplate).

This eliminates the dual retry paths in api_clients.py and makes model-level
cascade fallback explicit and testable.

Usage:
    from orchestrator.resilience import ResiliencePolicy, RetryTemplate, run_with_resilience

    policy = RetryTemplate.CODE_GEN.to_policy()
    result = await run_with_resilience([primary_fn, fallback_fn], policy)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from .circuit_breaker import CircuitBreakerOpen, CircuitBreakerRegistry
from .models import FALLBACK_CHAIN, Model, TaskType

logger = logging.getLogger("orchestrator.resilience")

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResiliencePolicy:
    """
    Immutable value object describing retry / fallback / timeout behavior.

    Attributes:
        retries:            Max retry attempts *per model* before falling back.
        timeout:            Per-request timeout in seconds.
        backoff_base:       Exponential backoff multiplier.
        backoff_max:        Cap on backoff wait time (seconds).
        jitter:             Add ±20 % random jitter to wait times.
        fallback_chain:     Ordered list of fallback models to try after primary
                            exhausts its retries.  None = no fallback.
        retryable_exceptions:
                            Tuple of exception types that trigger a retry.
    """

    retries: int = 2
    timeout: float = 60.0
    backoff_base: float = 2.0
    backoff_max: float = 30.0
    jitter: bool = True
    fallback_chain: tuple[Model, ...] | None = None
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
    )

    def with_fallback(self, *models: Model) -> ResiliencePolicy:
        """Return a new policy with the given fallback chain."""
        return ResiliencePolicy(
            retries=self.retries,
            timeout=self.timeout,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            jitter=self.jitter,
            fallback_chain=models,
            retryable_exceptions=self.retryable_exceptions,
        )


class RetryTemplate(Enum):
    """
    Task-aware retry presets.

    Each preset is tuned for the latency / determinism / cost profile of the
    task type.  Call ``to_policy()`` to materialise a ``ResiliencePolicy``.
    """

    DEFAULT = "default"
    CODE_GEN = "code_generation"
    CODE_REVIEW = "code_review"
    EVALUATE = "evaluation"
    DECOMPOSE = "decomposition"
    REASONING = "complex_reasoning"

    def to_policy(self) -> ResiliencePolicy:
        """Convert template to a concrete policy."""
        base = ResiliencePolicy()
        if self == RetryTemplate.CODE_GEN:
            return ResiliencePolicy(
                retries=3,
                timeout=120.0,
                backoff_base=2.0,
                backoff_max=30.0,
                jitter=True,
            )
        if self == RetryTemplate.CODE_REVIEW:
            return ResiliencePolicy(
                retries=2,
                timeout=60.0,
                backoff_base=2.0,
                backoff_max=20.0,
                jitter=True,
            )
        if self == RetryTemplate.EVALUATE:
            return ResiliencePolicy(
                retries=2,
                timeout=45.0,
                backoff_base=1.5,
                backoff_max=10.0,
                jitter=True,
            )
        if self == RetryTemplate.DECOMPOSE:
            return ResiliencePolicy(
                retries=3,
                timeout=90.0,
                backoff_base=2.0,
                backoff_max=30.0,
                jitter=True,
            )
        if self == RetryTemplate.REASONING:
            return ResiliencePolicy(
                retries=2,
                timeout=300.0,
                backoff_base=2.0,
                backoff_max=60.0,
                jitter=True,
            )
        return base

    @classmethod
    def for_task_type(cls, task_type: TaskType) -> ResiliencePolicy:
        """Return the preset policy for a given TaskType."""
        mapping: dict[TaskType, RetryTemplate] = {
            TaskType.CODE_GEN: cls.CODE_GEN,
            TaskType.CODE_REVIEW: cls.CODE_REVIEW,
            TaskType.EVALUATE: cls.EVALUATE,
            TaskType.REASONING: cls.REASONING,
            TaskType.WRITING: cls.DEFAULT,
            TaskType.DATA_EXTRACT: cls.DEFAULT,
            TaskType.SUMMARIZE: cls.DEFAULT,
        }
        template = mapping.get(task_type, cls.DEFAULT)
        return template.to_policy()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback chain resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_fallback_chain(model: Model, max_depth: int = 3) -> list[Model]:
    """
    Resolve the fallback chain for *model* by walking ``FALLBACK_CHAIN``.

    The static ``FALLBACK_CHAIN`` is single-step; this helper recursively
    resolves up to *max_depth* hops, stopping on cycles or missing entries.

    Returns:
        Ordered list of fallback models (empty if no fallbacks configured).
    """
    chain: list[Model] = []
    current: Model | None = model
    seen: set[Model] = {model}

    for _ in range(max_depth):
        nxt = FALLBACK_CHAIN.get(current)
        if nxt is None or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt

    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Execution helper
# ─────────────────────────────────────────────────────────────────────────────


async def run_with_resilience(
    callables: list[Callable[[], Awaitable[T]]],
    policy: ResiliencePolicy,
    model_ids: list[str] | None = None,
    registry: CircuitBreakerRegistry | None = None,
) -> T:
    """
    Execute *callables* in order, retrying each per *policy*.

    The first callable that succeeds (after its retries) determines the result.
    If all callables exhaust their retries, the last exception is re-raised.

    Args:
        callables:  Ordered list of async callables (primary first, then fallbacks).
        policy:     Resilience policy governing retries and backoff.
        model_ids:  Optional list of model IDs parallel to *callables*.  When
                    provided together with *registry*, each callable is wrapped
                    in its per-model circuit breaker — tripped models are skipped
                    immediately without wasting retries or budget.
        registry:   Per-model CircuitBreakerRegistry.  If None, no circuit-breaker
                    gating is applied (backwards-compatible).

    Returns:
        The result of the first successful callable.

    Raises:
        The last exception raised by the final callable after all retries.
    """
    if not callables:
        raise ValueError("run_with_resilience requires at least one callable")

    # Build circuit-breaker-aware wrappers when registry is supplied
    if registry is not None and model_ids is not None:
        assert len(model_ids) == len(callables), (
            "model_ids must be the same length as callables"
        )
        breaker_callables: list[Callable[[], Awaitable[T]]] = []
        for fn, mid in zip(callables, model_ids):
            # Use the async-safe registry.get() to avoid the race condition in
            # get_sync() where two concurrent callers create duplicate breakers
            # for the same model_id, splitting failure counts across two objects.
            cb = await registry.get(mid)

            async def _wrapped(fn=fn, cb=cb) -> T:  # type: ignore[return]
                async with cb.context():
                    return await fn()

            breaker_callables.append(_wrapped)
        callables = breaker_callables

    # Lazy-import tenacity so this module loads even if tenacity is absent.
    try:
        from tenacity import (
            AsyncRetrying,
            before_sleep_log,
            retry_if_exception_type,
            stop_after_attempt,
            stop_after_delay,
            wait_exponential,
            wait_random,
        )
    except ImportError as exc:
        logger.warning("tenacity not available; running callables without retry: %s", exc)
        last_exc: Exception | None = None
        for fn in callables:
            try:
                return await fn()
            except Exception as e:
                last_exc = e
        raise last_exc or RuntimeError("All resilience callables failed")

    # Build tenacity wait strategy
    wait_strategy = wait_exponential(
        multiplier=policy.backoff_base, min=1.0, max=policy.backoff_max
    )
    if policy.jitter:
        wait_strategy = wait_strategy + wait_random(
            min=1.0 * 0.2, max=policy.backoff_max * 0.2
        )

    last_error: Exception | None = None

    for idx, fn in enumerate(callables):
        retrying = AsyncRetrying(
            stop=stop_after_attempt(policy.retries) | stop_after_delay(int(policy.timeout)),
            wait=wait_strategy,
            retry=retry_if_exception_type(policy.retryable_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        try:
            async for attempt in retrying:
                with attempt:
                    return await fn()
        except CircuitBreakerOpen as exc:
            # Circuit is OPEN — skip immediately, no retries
            last_error = exc
            remaining = len(callables) - idx - 1
            if remaining > 0:
                logger.info(
                    "Callable %d/%d skipped (circuit OPEN: %s); "
                    "trying fallback (%d remaining)",
                    idx + 1,
                    len(callables),
                    exc,
                    remaining,
                )
            continue
        except Exception as exc:
            last_error = exc
            remaining = len(callables) - idx - 1
            if remaining > 0:
                logger.warning(
                    "Callable %d/%d failed after retries (%s); "
                    "trying fallback (%d remaining)",
                    idx + 1,
                    len(callables),
                    type(exc).__name__,
                    remaining,
                )
            continue

    # All callables exhausted
    raise last_error or RuntimeError("All resilience callables failed")


# ─────────────────────────────────────────────────────────────────────────────
# Event stub (for later UnifiedEventBus integration)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FallbackTriggeredEvent:
    """Lightweight event recording that a fallback model was used."""

    primary_model: str
    fallback_model: str
    attempt_number: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "attempt_number": self.attempt_number,
            "reason": self.reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cost-tier cascade policy
# ─────────────────────────────────────────────────────────────────────────────


class CostTier(Enum):
    """Ordered cost tiers for cascade fallback: FREE < BUDGET < PREMIUM."""

    FREE = "free"
    BUDGET = "budget"
    PREMIUM = "premium"


# Cost thresholds (USD per 1M input tokens) that define tier boundaries.
_TIER_THRESHOLDS: dict[CostTier, float] = {
    CostTier.FREE: 0.0,      # $0 — free models
    CostTier.BUDGET: 0.5,    # ≤ $0.50/1M tokens
    CostTier.PREMIUM: 999.0, # > $0.50/1M tokens
}


def classify_model_tier(model: Model) -> CostTier:
    """
    Return the cost tier of *model* by looking up COST_TABLE.

    Defaults to PREMIUM if the model is not in COST_TABLE (fail-safe:
    assume expensive rather than assume free).
    """
    from .models import COST_TABLE

    costs = COST_TABLE.get(model)
    if costs is None:
        return CostTier.PREMIUM
    input_cost = costs.get("input", 999.0)
    if input_cost <= 0.0:
        return CostTier.FREE
    if input_cost <= _TIER_THRESHOLDS[CostTier.BUDGET]:
        return CostTier.BUDGET
    return CostTier.PREMIUM


@dataclass(frozen=True)
class CascadePolicy:
    """
    Cost-tier–aware cascade policy: FREE → BUDGET → PREMIUM.

    Builds a fallback chain for *primary_model* that escalates through
    cost tiers only when cheaper options are exhausted or their circuit
    breakers are tripped.

    Usage:
        policy = CascadePolicy.for_model(Model.GPT_4O_MINI)
        chain = policy.ordered_chain
        # [GPT_4O_MINI, <budget_fallback>, <premium_fallback>]

    When paired with CircuitBreakerRegistry and run_with_resilience, tripped
    models are skipped automatically without wasting budget on retries.
    """

    primary_model: Model
    ordered_chain: tuple[Model, ...]

    @classmethod
    def for_model(cls, model: Model, max_depth: int = 3) -> "CascadePolicy":
        """
        Build a CascadePolicy for *model* using the static FALLBACK_CHAIN.

        The chain is sorted so that cheaper models come first — no point
        escalating to a premium model if a budget one would do.
        """
        raw_chain = resolve_fallback_chain(model, max_depth=max_depth)
        # Sort by cost tier (FREE < BUDGET < PREMIUM) while preserving
        # the original chain ordering within each tier.
        tier_order = {CostTier.FREE: 0, CostTier.BUDGET: 1, CostTier.PREMIUM: 2}
        sorted_chain = sorted(
            raw_chain,
            key=lambda m: tier_order[classify_model_tier(m)],
        )
        return cls(
            primary_model=model,
            ordered_chain=(model, *sorted_chain),
        )

    def to_policy(self, base: ResiliencePolicy | None = None) -> ResiliencePolicy:
        """Return a ResiliencePolicy with this cascade's fallback chain."""
        base = base or ResiliencePolicy()
        return ResiliencePolicy(
            retries=base.retries,
            timeout=base.timeout,
            backoff_base=base.backoff_base,
            backoff_max=base.backoff_max,
            jitter=base.jitter,
            fallback_chain=self.ordered_chain[1:],  # exclude primary
            retryable_exceptions=base.retryable_exceptions,
        )
