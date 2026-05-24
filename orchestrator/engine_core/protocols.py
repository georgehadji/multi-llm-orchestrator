"""
Protocols — Narrow role interfaces for service DI
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 6 of the Master Architecture Enhancement Plan.
Defines narrow Protocol types that collaborators receive instead
of the entire Orchestrator (self). This decouples services from
the engine's internal API, enabling testability and swapping.

Each Protocol captures exactly what a service needs from its
orchestration context — nothing more.

Usage:
    from .engine_core.protocols import ModelProvider, BudgetTracker

    class ExecutorService:
        def __init__(self, models: ModelProvider, budget: BudgetTracker):
            ...
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ..models import Model, Task, TaskResult, TaskType

logger = logging.getLogger("orchestrator.engine_core.protocols")


# ─────────────────────────────────────────────
# Model Provider
# ─────────────────────────────────────────────


@runtime_checkable
class ModelProvider(Protocol):
    """Provides model lists and health status.

    Satisfied by: Orchestrator (via _get_available_models + api_health)
    Used by: GeneratorService, GenerateStage, EvaluatorService, CritiqueStage
    """

    def get_available_models(self, task_type: TaskType | None = None) -> list[Model]:
        """Return list of healthy models for a given task type."""
        ...

    @property
    def api_health(self) -> dict[Model, bool]:
        """Per-model health status. True = currently usable."""
        ...


# ─────────────────────────────────────────────
# Budget Tracker
# ─────────────────────────────────────────────


@runtime_checkable
class BudgetTracker(Protocol):
    """Tracks and enforces budget constraints.

    Satisfied by: Orchestrator (via budget)
    Used by: ExecutorService, GenerateStage
    """

    async def reserve(self, amount: float, phase: str) -> str:
        """Reserve budget. Returns reservation ID."""
        ...

    async def commit_reservation(self, reservation_id: str, actual: float, phase: str) -> None:
        """Commit a previously reserved budget amount."""
        ...

    async def charge(self, amount: float, phase: str | None = None) -> bool:
        """Charge budget. Returns True if under budget."""
        ...

    @property
    def remaining(self) -> float:
        """Remaining budget in USD."""
        ...

    @property
    def max_usd(self) -> float:
        """Maximum budget in USD."""
        ...


# ─────────────────────────────────────────────
# Task Runner
# ─────────────────────────────────────────────


@runtime_checkable
class TaskRunner(Protocol):
    """Executes a single task.

    Satisfied by: Orchestrator._execute_task
    Used by: ExecutorService
    """

    async def execute_task(self, task: Task, policy: Any = None) -> TaskResult:
        """Execute a task and return the result."""
        ...


# ─────────────────────────────────────────────
# Context Provider
# ─────────────────────────────────────────────


@runtime_checkable
class ContextProvider(Protocol):
    """Provides project-level context for tasks.

    Satisfied by: Orchestrator
    Used by: ExecutorService, Decomposer
    """

    @property
    def project_context(self) -> Any:
        """Cross-phase project context accumulator."""
        ...

    @property
    def context_truncation_limit(self) -> int:
        """Maximum context length before truncation."""
        ...


# ─────────────────────────────────────────────
# Event Emitter
# ─────────────────────────────────────────────


@runtime_checkable
class EventEmitter(Protocol):
    """Fires hook events for observability.

    Satisfied by: Orchestrator._hook_registry
    Used by: ExecutorService, PreflightStage, TaskValidator
    """

    def fire(self, event_type: str, **kwargs: Any) -> None:
        """Fire a hook event with keyword arguments."""
        ...


# ─────────────────────────────────────────────
# Circuit Breaker Access
# ─────────────────────────────────────────────


@runtime_checkable
class CircuitBreakerAccess(Protocol):
    """Circuit breaker status and control for a model.

    Satisfied by: Orchestrator or CircuitBreakerRegistry
    Used by: GenerateStage, CritiqueStage
    """

    def is_model_available(self, model: Model) -> bool:
        """Check if a model's circuit breaker is closed."""
        ...

    def record_success(self, model: Model) -> None:
        """Record a successful call for a model."""
        ...

    def record_failure(self, model: Model) -> None:
        """Record a failed call for a model."""
        ...


# ─────────────────────────────────────────────
# Telemetry Recorder
# ─────────────────────────────────────────────


@runtime_checkable
class TelemetryRecorder(Protocol):
    """Records call metrics for observability.

    Satisfied by: ObservabilityService or TelemetryCollector
    Used by: ExecutorService, GenerateStage, EvaluateStage
    """

    def record_call(self, model: Model, latency: float, tokens: int, cost: float) -> None:
        """Record a model call."""
        ...

    def record_error(self, model: Model, error: str) -> None:
        """Record a model error."""
        ...

    def error_rate(self, model: Model) -> float:
        """Get error rate for a model (rolling window)."""
        ...


# ─────────────────────────────────────────────
# Logging Provider
# ─────────────────────────────────────────────


@runtime_checkable
class LoggingProvider(Protocol):
    """Provides structured logging capabilities.

    Satisfied by: logger instance or instrumentation layer
    """

    def debug(self, msg: str, *args: Any) -> None:
        ...

    def info(self, msg: str, *args: Any) -> None:
        ...

    def warning(self, msg: str, *args: Any) -> None:
        ...

    def error(self, msg: str, *args: Any) -> None:
        ...
