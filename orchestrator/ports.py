"""
Ports — abstract interfaces for the application layer.
=======================================================
Defines the boundary between the application core (engine.py, services/)
and the infrastructure adapters (cache.py, state.py, events/streaming).

WHY PORTS?
  engine.py currently imports concrete infrastructure classes directly:
    from .cache import DiskCache        # violates dependency rule
    from .state import StateManager     # violates dependency rule

  By depending on these Protocol definitions instead, the engine becomes
  testable with lightweight fakes, and infrastructure can be swapped
  (e.g. Redis cache, Postgres state) without touching application logic.

PATTERN: Structural subtyping (typing.Protocol)
  Concrete adapters satisfy the protocols implicitly — no changes needed
  to DiskCache, StateManager, or ProjectEventBus.  The runtime check is
  optional; mypy enforces it statically.

USAGE:
    # engine.py constructor — accepts either the real adapter or a fake:
    def __init__(
        self,
        cache: CachePort | None = None,
        state_manager: StatePort | None = None,
        ...
    ):
        self.cache: CachePort = cache or DiskCache()
        self.state_mgr: StatePort = state_manager or StateManager()

    # Tests — inject lightweight fakes:
    from orchestrator.ports import CachePort
    from unittest.mock import AsyncMock, MagicMock

    fake_cache = MagicMock(spec=CachePort)
    fake_cache.get = AsyncMock(return_value=None)
    fake_cache.put = AsyncMock()
    fake_cache.close = AsyncMock()
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import ProjectState


# ─────────────────────────────────────────────────────────────────────────────
# CachePort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CachePort(Protocol):
    """
    Read/write LLM response cache.

    Satisfied by: orchestrator.cache.DiskCache
    """

    async def get(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        system: str | None,
        temperature: float,
    ) -> Any | None:
        """
        Return a cached response, or None if not found / expired.

        Args:
            model_id:    The model identifier string (e.g. "openai/gpt-4o").
            prompt:      The user prompt text.
            max_tokens:  Max tokens requested (part of cache key).
            system:      Optional system prompt.
            temperature: Sampling temperature (part of cache key).

        Returns:
            Cached response object (type depends on adapter), or None.
        """
        ...

    async def put(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        system: str | None,
        temperature: float,
        response: Any,
    ) -> None:
        """
        Store *response* under the given key.

        Implementations may apply TTL, size limits, or compression; callers
        are not responsible for cache eviction policy.
        """
        ...

    async def close(self) -> None:
        """Release any held resources (connections, file handles, etc.)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# StatePort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class StatePort(Protocol):
    """
    Persistent project state store.

    Satisfied by: orchestrator.state.StateManager
    """

    async def save_project(self, project_id: str, state: ProjectState) -> None:
        """
        Persist *state* for *project_id*, replacing any existing record.

        Must be durable: a crash after this call returns must not lose the state.
        """
        ...

    async def load_project(self, project_id: str) -> ProjectState | None:
        """
        Load the most recent state for *project_id*.

        Returns None if the project has never been saved, or if the persisted
        state fails schema validation (corrupted / migrated away from).
        """
        ...

    async def save_checkpoint(
        self, project_id: str, task_id: str, state: ProjectState
    ) -> None:
        """
        Save a mid-run checkpoint for crash recovery.

        Checkpoints are lighter-weight than full saves and may be pruned by
        the adapter.  Callers treat them as best-effort.
        """
        ...

    async def close(self) -> None:
        """Flush and release any held resources."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# EventPort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class EventPort(Protocol):
    """
    Async event bus for streaming project progress to external consumers.

    Satisfied by: orchestrator.streaming.ProjectEventBus
    """

    async def publish(self, event: Any) -> None:
        """
        Publish *event* to all current subscribers.

        Must not raise even if there are zero subscribers.
        Implementations should handle slow consumers without blocking the
        caller (e.g. via asyncio.Queue with overflow policy).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# NullAdapters — lightweight no-op implementations for testing / dry-runs
# ─────────────────────────────────────────────────────────────────────────────


class NullCache:
    """
    No-op cache adapter.  Every get() misses; put() is a no-op.

    Useful for unit tests that should not hit disk/sqlite and for dry-run
    orchestration where caching is intentionally disabled.
    """

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
        system: str | None,
        temperature: float,
        response: Any,
    ) -> None:
        pass

    async def close(self) -> None:
        pass


class NullState:
    """
    In-memory state adapter.  Stores state in a dict; not durable across runs.

    Useful for unit tests and short-lived orchestrator instances that don't
    need crash recovery.
    """

    def __init__(self) -> None:
        self._store: dict[str, ProjectState] = {}
        self._checkpoints: dict[str, ProjectState] = {}

    async def save_project(self, project_id: str, state: ProjectState) -> None:
        self._store[project_id] = state

    async def load_project(self, project_id: str) -> ProjectState | None:
        return self._store.get(project_id)

    async def save_checkpoint(
        self, project_id: str, task_id: str, state: ProjectState
    ) -> None:
        self._checkpoints[f"{project_id}:{task_id}"] = state

    async def close(self) -> None:
        pass


class NullEventBus:
    """
    No-op event bus.  publish() discards all events.

    Useful for tests that don't need streaming progress output.
    """

    async def publish(self, event: Any) -> None:
        pass
