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
    async def save_circuit_breaker_state(self, model_name: str, failure_count: int) -> None: ...
    async def load_circuit_breaker_state(self) -> dict[str, int]: ...
    async def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# EventPort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class EventPort(Protocol):
    """Async event bus. Satisfied by ProjectEventBus."""

    async def publish(self, event: Any) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# ConfigPort
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ConfigPort(Protocol):
    """Architectural configuration store. Satisfied by ConfigAdapter."""

    def get_costs(self) -> dict[str, dict[str, float]]: ...
    def get_routing(self) -> dict[str, list[str]]: ...
    def get_fallbacks(self) -> dict[str, str]: ...
    def get_thresholds(self) -> dict[str, float]: ...
    def get_limits(self) -> dict[str, int]: ...


# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class LLMClient(Protocol):
    """Minimal async LLM call interface for application-layer services.
    Satisfied by: orchestrator.api_clients.UnifiedClient
    """
    async def call(self, model, prompt, system="", max_tokens=1500, temperature=0.3, timeout=120, **kwargs): ...



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

    async def save_circuit_breaker_state(self, model_name: str, failure_count: int) -> None:
        pass  # NullState doesn't persist circuit breaker state

    async def load_circuit_breaker_state(self) -> dict[str, int]:
        return {}

    async def close(self) -> None:
        pass


class NullEventBus:
    """No-op event bus. publish() discards all events."""

    async def publish(self, event: Any) -> None:
        pass
