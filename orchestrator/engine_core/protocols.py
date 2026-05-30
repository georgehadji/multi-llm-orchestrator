"""
Structural subtyping protocols for the engine core.
====================================================
Defines runtime-checkable Protocol interfaces that decouple the engine
from concrete implementations. Satisfying a protocol requires no explicit
subclassing — any object with the required attributes/methods qualifies.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Provides model availability and health information."""

    def get_available_models(self, task_type: Any = None) -> list[Any]: ...

    api_health: dict[Any, bool]


@runtime_checkable
class BudgetTracker(Protocol):
    """Tracks and enforces spend limits for a project run."""

    async def reserve(self, amount: float, task_id: str) -> bool: ...

    async def commit_reservation(self, task_id: str) -> None: ...

    async def charge(self, amount: float) -> None: ...

    @property
    def remaining(self) -> float: ...

    @property
    def max_usd(self) -> float: ...


@runtime_checkable
class TaskRunner(Protocol):
    """Executes a single task through the pipeline."""

    async def execute_task(self, task: Any, policy: Any = None) -> Any: ...


@runtime_checkable
class EventEmitter(Protocol):
    """Fires synchronous lifecycle events."""

    def fire(self, event: Any, *args: Any, **kwargs: Any) -> None: ...
