"""
Domain Ports — Abstract Interfaces (Protocols)
================================================
Defines the boundary between the application core and infrastructure adapters.

Satisfied by: orchestrator.infrastructure.cache.DiskCache (implements CachePort)
             orchestrator.infrastructure.state.StateManager (implements StatePort)

Pattern: Structural subtyping via typing.Protocol
  Concrete adapters satisfy protocols implicitly — no ABC registration needed.

NullAdapters for testing:
  NullCache   — every get() misses, put() is a no-op
  NullState   — in-memory dict-based store
  NullEventBus — discards all published events
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import ProjectState

# ─────────────────────────────────────────────────────────────────────────────
# CachePort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CachePort(Protocol):
    """Read/write LLM response cache. Satisfied by DiskCache."""

    async def get(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        system: str | None,
        temperature: float,
    ) -> Any | None: ...

    async def put(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        response: Any,
        tokens_input: int,
        tokens_output: int,
        system: str | None,
        temperature: float,
    ) -> None: ...

    async def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# StatePort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class StatePort(Protocol):
    """Persistent project state store. Satisfied by StateManager."""

    async def save_project(self, project_id: str, state: ProjectState) -> None: ...
    async def load_project(self, project_id: str) -> ProjectState | None: ...
    async def save_checkpoint(self, project_id: str, task_id: str, state: ProjectState) -> None: ...
    async def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# EventPort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class EventPort(Protocol):
    """Async event bus. Satisfied by ProjectEventBus."""

    async def publish(self, event: Any) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# NullAdapters — lightweight no-op implementations for testing
# ─────────────────────────────────────────────────────────────────────────────


class NullCache:
    """No-op cache. Every get() misses, put() is a no-op."""

    async def get(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        system: str | None,
        temperature: float,
    ) -> None:
        return None

    async def put(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        response: Any,
        tokens_input: int,
        tokens_output: int,
        system: str | None,
        temperature: float,
    ) -> None:
        pass

    async def close(self) -> None:
        pass


class NullState:
    """In-memory state store. Not durable across runs."""

    def __init__(self) -> None:
        self._store: dict[str, ProjectState] = {}
        self._checkpoints: dict[str, ProjectState] = {}

    async def save_project(self, project_id: str, state: ProjectState) -> None:
        self._store[project_id] = state

    async def load_project(self, project_id: str) -> ProjectState | None:
        return self._store.get(project_id)

    async def save_checkpoint(self, project_id: str, task_id: str, state: ProjectState) -> None:
        self._checkpoints[f"{project_id}:{task_id}"] = state

    async def close(self) -> None:
        pass


class NullEventBus:
    """No-op event bus. publish() discards all events."""

    async def publish(self, event: Any) -> None:
        pass
