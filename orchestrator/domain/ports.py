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

from dataclasses import dataclass
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
# TaskQueuePort  (work-queue abstraction for horizontal scaling)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class TaskQueuePort(Protocol):
    """Async work-queue for task dispatch and execution.

    Satisfied by: orchestrator.kanban.board.KanbanBoard
    Enables swapping the queue backend (Kanban → Redis, RabbitMQ, etc.)
    without changing dispatcher or engine code.
    """

    async def enqueue(self, project_spec: Any, priority: int = 0) -> str: ...
    async def claim_next(self, assignee: str) -> Any | None: ...
    async def complete(self, task_id: str, result: Any | None = None) -> bool: ...
    async def record_failure(self, task_id: str, error: str = "") -> bool: ...
    async def list_tasks(self, status: str | None = None, limit: int = 50) -> list[Any]: ...
    async def get_stats(self) -> dict[str, Any]: ...
    async def update_status(self, task_id: str, status: str, result: Any | None = None) -> bool: ...


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


# ─────────────────────────────────────────────────────────────────────────────
# LSPValidatorPort  (CodeWhale Phase 1 — deterministic post-generation validation)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LSPDiagnostic:
    """A single diagnostic from a language server (pyright, tsc, gopls, etc.)."""

    severity: str = "error"  # "error" | "warning" | "information" | "hint"
    message: str = ""
    line: int = 0  # 1-indexed
    column: int = 0  # 1-indexed
    source: str = ""  # e.g. "pyright", "tsc"
    code: str = ""  # e.g. "reportUndefinedVariable"


# ── Domain-layer utility functions for LSP diagnostics ────────────────────────
# These operate only on LSPDiagnostic (a domain type) and str/list,
# so they belong in the domain layer — not infrastructure.


def lsp_diagnostics_summary(diags: list[LSPDiagnostic]) -> str:
    """Build a human-readable summary of diagnostics for prompt injection."""
    if not diags:
        return ""

    errors = [d for d in diags if d.severity == "error"]
    warnings = [d for d in diags if d.severity == "warning"]

    parts: list[str] = []
    if errors:
        parts.append(f"### {len(errors)} Error(s)")
        for e in errors[:10]:
            code_str = f" ({e.code})" if e.code else ""
            parts.append(f"- L{e.line}:{e.column} {e.message}{code_str}")
        if len(errors) > 10:
            parts.append(f"- ... and {len(errors) - 10} more errors")

    if warnings:
        parts.append(f"### {len(warnings)} Warning(s)")
        for w in warnings[:10]:
            code_str = f" ({w.code})" if w.code else ""
            parts.append(f"- L{w.line}:{w.column} {w.message}{code_str}")
        if len(warnings) > 10:
            parts.append(f"- ... and {len(warnings) - 10} more warnings")

    return "\n".join(parts)


def lsp_inject_inline_diagnostics(
    code: str, diags: list[LSPDiagnostic], language: str = "python"
) -> str:
    """Inject diagnostics as inline comments above the flagged lines.

    Uses language-appropriate comment prefix (# for Python, // for TS/JS).
    """
    lines = code.splitlines()
    sorted_diags = sorted(diags, key=lambda d: d.line, reverse=True)

    # Comment prefix based on explicit language parameter
    prefix = "#" if language in ("python", "ruby", "bash", "shell") else "//"

    for d in sorted_diags:
        if d.severity not in ("error", "warning"):
            continue
        idx = max(0, min(d.line - 1, len(lines) - 1))
        code_str = f" ({d.code})" if d.code else ""
        comment = f"{prefix} LSP [{d.severity.upper()}]: {d.message}{code_str}"
        lines.insert(idx, comment)

    return "\n".join(lines)


@runtime_checkable
class LSPValidatorPort(Protocol):
    """Validates generated code via language server diagnostics.

    Satisfied by: orchestrator.infrastructure.lsp_validator.LspValidator
    Application layer (CritiqueCycle) imports this protocol, not the adapter.
    """

    async def validate(
        self, code: str, language: str = "python", filename: str = ""
    ) -> list[LSPDiagnostic]:
        """Validate a code string, return diagnostics.
        Writes to tempfile, runs language server, cleans up.
        """
        ...

    async def validate_file(self, filepath: str) -> list[LSPDiagnostic]:
        """Validate a file already on disk, return diagnostics."""
        ...

    def available_servers(self) -> frozenset[str]:
        """Return language IDs (e.g. 'python', 'typescript') for which a
        server binary is installed and executable.
        """
        ...


class NullLspValidator:
    """No-op fallback when LSP validation is disabled or no servers installed."""

    async def validate(
        self, code: str, language: str = "python", filename: str = ""
    ) -> list[LSPDiagnostic]:
        return []

    async def validate_file(self, filepath: str) -> list[LSPDiagnostic]:
        return []

    def available_servers(self) -> frozenset[str]:
        return frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# SnapshotPort  (CodeWhale Phase 2 — content-preserving workspace snapshots)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SnapshotPort(Protocol):
    """Content-preserving workspace snapshots.

    Unlike CheckpointManager (which stores metadata/hashes only), this port
    stores actual file contents — enabling true `rollback()` that restores
    file state, not just metadata inspection.

    Satisfied by: orchestrator.infrastructure.snapshot_store.GitSnapshotStore
                  orchestrator.infrastructure.snapshot_store.TarSnapshotStore
    """

    async def create(
        self,
        label: str,
        source_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a snapshot of source_dir, return snapshot_id.

        Args:
            label: Human-readable name (e.g. "before-refactor-main").
            source_dir: Directory whose contents to snapshot.
            metadata: Optional extra data to associate (project_id, budget, etc.).

        Returns:
            String snapshot_id (git SHA, tariff name, etc.).
        """
        ...

    async def restore(self, snapshot_id: str, target_dir: str) -> bool:
        """Restore file contents from snapshot into target_dir.

        Args:
            snapshot_id: ID returned by create().
            target_dir: Directory to restore into (created if missing).

        Returns:
            True if successful.
        """
        ...

    async def list_snapshots(
        self,
    ) -> list[dict[str, Any]]:
        """Return all snapshots with metadata.

        Returns list of dicts with keys:
            id, label, timestamp, file_count, total_size_bytes, metadata
        Sorted by timestamp descending (most recent first).
        """
        ...

    async def diff(self, snapshot_a: str, snapshot_b: str) -> dict[str, Any]:
        """Compare two snapshots.

        Returns dict with keys:
            added_files, removed_files, modified_files, file_diffs
        File_diffs is a dict path -> unified diff string (line-level).
        """
        ...

    async def delete(self, snapshot_id: str) -> bool:
        """Remove a snapshot and its stored content."""
        ...


class NullSnapshotStore:
    """No-op fallback when snapshot storage is disabled."""

    async def create(self, label, source_dir, metadata=None):
        return ""

    async def restore(self, snapshot_id, target_dir):
        return False

    async def list_snapshots(self):
        return []

    async def diff(self, snapshot_a, snapshot_b):
        return {"added_files": [], "removed_files": [], "modified_files": [], "file_diffs": {}}

    async def delete(self, snapshot_id):
        return False
