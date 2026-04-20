"""
Concurrency Budget Controller — Prevent budget overspend via concurrent jobs
=============================================================================
Author: Senior Distributed Systems Architect

CRITICAL FIX: Prevents race condition where multiple concurrent jobs both pass
budget check and then both charge, exceeding budget.

PROBLEM:
    Job A checks budget: $50 available
    Job B checks budget: $50 available
    Job A charges: $50 spent
    Job B charges: $50 spent
    Total: $100 spent (budget was $50!)

SOLUTION:
    Global concurrency budget semaphore with pessimistic reservation.
    Atomic check-and-reserve operation.

USAGE:
    from orchestrator.concurrency_controller import ConcurrencyBudget

    budget = ConcurrencyBudget(max_concurrent_jobs=5, max_concurrent_cost=100.0)

    async with budget.acquire(job_id="job_1", estimated_cost=20.0):
        # Guaranteed: budget reserved, no race condition
        result = await run_job()
        budget.charge(actual_cost=18.5)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("orchestrator.concurrency")


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency budget."""

    active_jobs: int
    available_slots: int
    total_reserved_usd: float
    total_spent_usd: float
    total_jobs_completed: int
    total_jobs_rejected: int


class ConcurrencyBudget:
    """
    Global concurrency budget controller.

    GUARANTEES:
    1. At most max_concurrent_jobs running simultaneously
    2. Total reserved cost <= max_concurrent_cost_usd
    3. Atomic check-and-reserve (no race condition)
    4. Automatic release on failure

    EXAMPLE:
        budget = ConcurrencyBudget(max_concurrent_jobs=5, max_concurrent_cost=100.0)

        async with budget.acquire(job_id="job_1", estimated_cost=20.0):
            # Budget slot acquired
            result = await expensive_operation()
            budget.charge("job_1", actual_cost=18.5)
            # Slot released automatically
    """

    def __init__(self, max_concurrent_jobs: int = 10, max_concurrent_cost_usd: float = 100.0):
        """
        Initialize concurrency budget.

        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs
            max_concurrent_cost_usd: Maximum total cost of concurrent jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_cost_usd = max_concurrent_cost_usd

        # Semaphores for limiting concurrency
        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        # Cost semaphore uses cents for integer precision
        self._cost_semaphore = asyncio.Semaphore(int(max_concurrent_cost_usd * 100))

        # Track active jobs
        self._active_jobs: dict[str, float] = {}  # job_id -> reserved_cost
        self._lock = asyncio.Lock()

        # Statistics
        self._total_reserved: float = 0.0
        self._total_spent: float = 0.0
        self._jobs_completed: int = 0
        self._jobs_rejected: int = 0

    @asynccontextmanager
    async def acquire(self, job_id: str, estimated_cost: float, timeout_seconds: float = 30.0):
        """
        Acquire budget slot for a job.

        This is an atomic check-and-reserve operation. If successful,
        the budget is guaranteed to be available for this job.

        Args:
            job_id: Unique identifier for the job
            estimated_cost: Estimated cost in USD
            timeout_seconds: Maximum time to wait for budget

        Yields:
            self (for chaining operations)

        Raises:
            TimeoutError: If budget not available within timeout
        """
        # Cost is tracked in integer cents so the semaphore unit = 1 cent.
        # We MUST save this value and acquire/release exactly cost_cents units —
        # previously this line computed the value and immediately discarded it,
        # so every job acquired exactly 1 cent regardless of estimated_cost.
        cost_cents = max(1, int(estimated_cost * 100))
        job_acquired = False
        cost_acquired = 0          # tracks how many cent-units we hold (for safe release)
        start_time = time.monotonic()

        try:
            # Acquire job slot
            try:
                await asyncio.wait_for(self._job_semaphore.acquire(), timeout=timeout_seconds)
                job_acquired = True
            except asyncio.TimeoutError:
                self._jobs_rejected += 1
                raise TimeoutError(
                    f"Could not acquire job slot for {job_id} within {timeout_seconds}s "
                    f"(max_concurrent_jobs={self.max_concurrent_jobs})"
                )

            # Acquire cost_cents units one-by-one (asyncio.Semaphore has no batch acquire)
            try:
                for _ in range(cost_cents):
                    await asyncio.wait_for(
                        self._cost_semaphore.acquire(),
                        timeout=max(0.1, timeout_seconds - (time.monotonic() - start_time)),
                    )
                    cost_acquired += 1
            except asyncio.TimeoutError:
                self._jobs_rejected += 1
                raise TimeoutError(
                    f"Could not acquire cost budget for {job_id} within {timeout_seconds}s "
                    f"(estimated_cost=${estimated_cost:.2f}, available=${self.available_budget:.2f})"
                )

            # Record reservation
            async with self._lock:
                self._active_jobs[job_id] = estimated_cost
                self._total_reserved += estimated_cost

            logger.debug(f"Budget acquired for {job_id}: ${estimated_cost:.2f}")

            yield self

        finally:
            # Release exactly the units we acquired (handles partial acquisition on timeout)
            for _ in range(cost_acquired):
                self._cost_semaphore.release()
            if job_acquired:
                self._job_semaphore.release()

            # Clean up reservation
            async with self._lock:
                if job_id in self._active_jobs:
                    reserved = self._active_jobs.pop(job_id)
                    self._total_reserved -= reserved

    def charge(self, job_id: str, actual_cost: float) -> None:
        """
        Record actual cost spent.

        This is informational only - the budget was already reserved.
        Call this after job completion for accurate statistics.

        Args:
            job_id: Job identifier
            actual_cost: Actual cost in USD
        """

        async def _record():
            async with self._lock:
                self._total_spent += actual_cost
                self._jobs_completed += 1
                logger.debug(f"Job {job_id} charged: ${actual_cost:.2f}")

        # Schedule charge (fire-and-forget)
        asyncio.create_task(_record())

    def release(self, job_id: str) -> None:
        """
        Explicitly release budget for a job.

        Usually not needed - budget is released automatically when
        exiting the context manager. Use this for manual cleanup.

        Args:
            job_id: Job identifier
        """

        async def _release():
            async with self._lock:
                if job_id in self._active_jobs:
                    reserved = self._active_jobs.pop(job_id)
                    self._total_reserved -= reserved
                    logger.debug(f"Budget released for {job_id}: ${reserved:.2f}")

        asyncio.create_task(_release())

    @property
    def available_slots(self) -> int:
        """Number of available job slots."""
        return self._job_semaphore._value

    @property
    def available_budget(self) -> float:
        """Available cost budget in USD."""
        return self._cost_semaphore._value / 100.0

    @property
    def active_jobs(self) -> dict[str, float]:
        """Currently active jobs and their reserved costs."""
        return dict(self._active_jobs)

    def get_stats(self) -> ConcurrencyStats:
        """Get current statistics."""
        return ConcurrencyStats(
            active_jobs=len(self._active_jobs),
            available_slots=self.available_slots,
            total_reserved_usd=self._total_reserved,
            total_spent_usd=self._total_spent,
            total_jobs_completed=self._jobs_completed,
            total_jobs_rejected=self._jobs_rejected,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dict."""
        return {
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_concurrent_cost_usd": self.max_concurrent_cost_usd,
            "available_slots": self.available_slots,
            "available_budget": self.available_budget,
            "active_jobs": self.active_jobs,
            "total_reserved_usd": self._total_reserved,
            "total_spent_usd": self._total_spent,
            "jobs_completed": self._jobs_completed,
            "jobs_rejected": self._jobs_rejected,
        }


class GlobalConcurrencyController:
    """
    Singleton controller for global concurrency.

    Ensures all orchestrator instances share the same budget limits.
    Use this when running multiple Orchestrator instances.

    USAGE:
        controller = GlobalConcurrencyController.get_instance()

        async with controller.budget.acquire(job_id, estimated_cost):
            await run_job()
    """

    _instance: GlobalConcurrencyController | None = None
    _lock = asyncio.Lock()

    def __init__(self, max_concurrent_jobs: int = 10, max_concurrent_cost_usd: float = 100.0):
        self.budget = ConcurrencyBudget(
            max_concurrent_jobs=max_concurrent_jobs, max_concurrent_cost_usd=max_concurrent_cost_usd
        )

    @classmethod
    async def get_instance(cls) -> GlobalConcurrencyController:
        """Get singleton instance (async-safe)."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> GlobalConcurrencyController:
        """Get singleton instance (sync, for backward compatibility)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


class RateLimitedConcurrencyBudget(ConcurrencyBudget):
    """
    Concurrency budget with per-tenant rate limiting.

    Combines job/cost limiting with TPM/RPM rate limits.
    """

    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        max_concurrent_cost_usd: float = 100.0,
        default_tpm: int = 100000,
        default_rpm: int = 1000,
    ):
        super().__init__(max_concurrent_jobs, max_concurrent_cost_usd)

        self._default_tpm = default_tpm
        self._default_rpm = default_rpm

        # Per-tenant rate limits
        self._tenant_limits: dict[str, dict[str, int]] = {}
        self._tenant_usage: dict[str, dict[str, int]] = {}

    def set_tenant_limits(
        self, tenant: str, tpm: int | None = None, rpm: int | None = None
    ) -> None:
        """Set rate limits for a tenant."""
        if tenant not in self._tenant_limits:
            self._tenant_limits[tenant] = {}

        if tpm is not None:
            self._tenant_limits[tenant]["tpm"] = tpm
        if rpm is not None:
            self._tenant_limits[tenant]["rpm"] = rpm

    def check_rate_limit(self, tenant: str, tokens: int) -> bool:
        """
        Check if request is within rate limits.

        Args:
            tenant: Tenant identifier
            tokens: Number of tokens in request

        Returns:
            True if within limits, False otherwise
        """
        limits = self._tenant_limits.get(
            tenant, {"tpm": self._default_tpm, "rpm": self._default_rpm}
        )

        usage = self._tenant_usage.get(tenant, {"tokens": 0, "requests": 0})

        if usage["tokens"] + tokens > limits.get("tpm", self._default_tpm):
            return False

        return not usage["requests"] + 1 > limits.get("rpm", self._default_rpm)

    def record_usage(self, tenant: str, tokens: int) -> None:
        """Record token usage for rate limiting."""
        if tenant not in self._tenant_usage:
            self._tenant_usage[tenant] = {"tokens": 0, "requests": 0}

        self._tenant_usage[tenant]["tokens"] += tokens
        self._tenant_usage[tenant]["requests"] += 1


# Convenience function
def get_concurrency_budget() -> ConcurrencyBudget:
    """Get global concurrency budget."""
    return GlobalConcurrencyController.get_instance_sync().budget


# ─────────────────────────────────────────────────────────────────────────────
# Task-level concurrency guard
# ─────────────────────────────────────────────────────────────────────────────


class TaskConcurrencyGuard:
    """
    Named, observable async semaphore for task-level execution control.

    Formalises the anonymous ``asyncio.Semaphore`` that previously lived inside
    ``engine._execute_all``.  Injected into ``ExecutorService`` so task
    parallelism can be controlled, observed, and tested independently.

    Args:
        name:           Human-readable identifier used in logs.
        max_concurrent: Slots available simultaneously.  Defaults to 1
                        (serial execution) — the safe baseline until
                        shared-state safety is fully proven.

    Usage:
        guard = TaskConcurrencyGuard(name="tasks", max_concurrent=1)
        async with guard:
            result = await executor.execute(task)
    """

    def __init__(self, name: str = "tasks", max_concurrent: int = 1) -> None:
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: int = 0
        self._waiting: int = 0
        self._total_acquired: int = 0
        self._total_wait_ms: float = 0.0
        self._stat_lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def waiting_count(self) -> int:
        return self._waiting

    async def __aenter__(self) -> "TaskConcurrencyGuard":
        async with self._stat_lock:
            self._waiting += 1
        t0 = time.monotonic()
        await self._semaphore.acquire()
        wait_ms = (time.monotonic() - t0) * 1000
        async with self._stat_lock:
            self._waiting -= 1
            self._active += 1
            self._total_acquired += 1
            self._total_wait_ms += wait_ms
        if wait_ms > 100:
            logger.debug(
                "guard '%s': waited %.0fms (active=%d/%d waiting=%d)",
                self.name, wait_ms, self._active, self.max_concurrent, self._waiting,
            )
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._semaphore.release()
        async with self._stat_lock:
            self._active -= 1

    def stats(self) -> dict[str, Any]:
        avg_wait = (
            self._total_wait_ms / self._total_acquired if self._total_acquired else 0.0
        )
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "waiting": self._waiting,
            "total_acquired": self._total_acquired,
            "avg_wait_ms": round(avg_wait, 1),
        }
