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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..models import ProjectState

if TYPE_CHECKING:
    from ..models import Model, TaskType

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

    async def call(  # type: ignore[no-untyped-def]
        self, model, prompt, system="", max_tokens=1500, temperature=0.3, timeout=120, **kwargs
    ): ...


# ─────────────────────────────────────────────────────────────────────────────
# Application-layer service ports (P2-1)
# These are satisfied by the concrete service classes wired in ServiceContainer.
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class PlannerPort(Protocol):
    """Model selection service. Satisfied by ModelSelector."""

    def available_models(self, task_type: TaskType) -> list[Model]: ...
    def select(self, task_type: TaskType) -> Model | None: ...


@runtime_checkable
class TelemetryPort(Protocol):
    """Metrics recorder. Satisfied by TelemetryCollector."""

    def record_call(
        self,
        model: Model,
        latency_ms: float,
        cost_usd: float,
        success: bool = True,
    ) -> None: ...


@runtime_checkable
class PolicyEnginePort(Protocol):
    """Policy evaluation. Satisfied by PolicyEngine."""

    def evaluate(self, job_spec: Any, profile: Any) -> Any: ...


@runtime_checkable
class HookRegistryPort(Protocol):
    """Synchronous lifecycle hook dispatch. Satisfied by HookRegistry."""

    def fire(self, event_type: Any, **kwargs: Any) -> None: ...
    def add(self, event: Any, callback: Any) -> None: ...


@runtime_checkable
class ValidatorPort(Protocol):
    """Task output validation. Satisfied by TaskValidator."""

    async def validate(self, task: Any, output: str) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────────
# NullAdapters — lightweight no-op implementations for testing
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SkillStorePort  (SkillOpt — self-improving skill system)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SkillStorePort(Protocol):
    """Persistence layer for skill documents and training trajectories.

    Satisfied by: orchestrator.application.skill_store.SkillStore
    """

    async def save_trajectory(self, t: Any) -> None: ...
    async def load_trajectories(self, task_type: Any, limit: int = 50) -> list[Any]: ...
    async def save_skill(
        self, task_type: Any, skill_doc: str, score: float, epoch: int
    ) -> None: ...
    async def load_best_skill(self, task_type: Any) -> tuple[str, float, int] | None: ...
    async def save_negative_feedback(
        self, task_type: Any, patches: list[Any], reason: str
    ) -> None: ...
    async def load_negative_feedback(self, task_type: Any, limit: int = 20) -> list[dict]: ...  # type: ignore[type-arg]
    async def close(self) -> None: ...


class NullSkillStore:
    """No-op SkillStore for testing. All writes are discarded."""

    async def save_trajectory(self, t: Any) -> None:
        pass

    async def load_trajectories(self, task_type: Any, limit: int = 50) -> list[Any]:
        return []

    async def save_skill(self, task_type: Any, skill_doc: str, score: float, epoch: int) -> None:
        pass

    async def load_best_skill(self, task_type: Any) -> None:
        return None

    async def save_patches(
        self, task_type: Any, epoch: int, patches: list[Any], accepted: bool
    ) -> None:
        pass

    async def save_negative_feedback(self, task_type: Any, patches: list[Any], reason: str) -> None:
        pass

    async def load_negative_feedback(self, task_type: Any, limit: int = 20) -> list[dict]:  # type: ignore[type-arg]
        return []

    async def close(self) -> None:
        pass


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


class NullHookRegistry:
    """No-op hook registry. fire() and add() are silent no-ops."""

    def fire(self, event_type: Any, **kwargs: Any) -> None:
        pass

    def add(self, event: Any, callback: Any) -> None:
        pass


class NullEventBus:
    """No-op event bus. publish() discards all events."""

    async def publish(self, event: Any) -> None:
        pass
