"""
Memory Provider — Pluggable Memory Backend ABC
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Domain-specific plugin surface for memory backends. Implementations
live in ~/.orchestrator/plugins/memory/<name>/__init__.py or bundled
under orchestrator/plugins/memory/<name>/.

Follows Hermes Agent's MemoryProvider pattern: providers implement
sync_turn() for post-task persistence and prefetch() for pre-task
context retrieval.

BuiltinMemoryProvider wraps the existing telemetry_store + bm25_search
infrastructure. Third-party providers (Honcho, Mem0, etc.) implement
the ABC and register via PluginManager.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import TaskResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryProviderMetadata:
    """Metadata about a memory provider plugin."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""


@dataclass
class MemoryQueryResult:
    """Single result from a memory provider prefetch."""

    content: str
    score: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryProvider ABC
# ─────────────────────────────────────────────────────────────────────────────


class MemoryProvider(ABC):
    """Pluggable memory backend.

    Each provider is responsible for:
    - Persisting task results after each task (sync_turn)
    - Returning relevant context before a task starts (prefetch)
    - Cleaning up on shutdown (shutdown)

    Lifecycle:
        1. provider.initialize(hermes_home)  — on registration
        2. provider.prefetch(query)           — before each task
        3. provider.sync_turn(task_id, result) — after each task
        4. provider.shutdown()                — on orchestrator shutdown
    """

    metadata: MemoryProviderMetadata

    @abstractmethod
    async def initialize(self, orchestrator_home: Path) -> None:
        """Initialize provider resources.

        Args:
            orchestrator_home: Path to ~/.orchestrator_cache/ or equivalent.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release provider resources. Called at orchestrator shutdown."""
        ...

    @abstractmethod
    async def sync_turn(
        self,
        task_id: str,
        task_type: str,
        result: TaskResult,
    ) -> None:
        """Persist a completed task result.

        Called after each task completes. Fire-and-forget by default.

        Args:
            task_id: Unique task identifier.
            task_type: String task type (e.g. "code_generation").
            result: The completed TaskResult with score, cost, etc.
        """
        ...

    @abstractmethod
    async def prefetch(self, query: str, limit: int = 5) -> list[MemoryQueryResult]:
        """Return relevant context for an upcoming task.

        Args:
            query: The task prompt or project description to match against.
            limit: Maximum number of results to return.

        Returns:
            List of MemoryQueryResult, ordered by relevance descending.
            Empty list when no relevant context is available.
        """
        ...

    async def post_setup(self, config: dict[str, Any] | None = None) -> None:
        """Optional: post-setup-wizard integration.

        Called after the provider is registered and initialized.  Providers
        can use this to prompt for missing config values or run a health check.

        Args:
            config: Configuration dict from config.yaml (memory.<name>.*).
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────
# BuiltinMemoryProvider — wraps telemetry_store + BM25 search
# ─────────────────────────────────────────────────────────────────────────────


class BuiltinMemoryProvider(MemoryProvider):
    """Default memory provider using the existing telemetry + BM25 infrastructure.

    Stores:
    - Task results as routing_events in TelemetryStore
    - Full text in BM25 search for cross-project recall

    Prefetch queries BM25 with the task prompt and returns matching
    prior task outputs as context.
    """

    def __init__(self) -> None:
        self.metadata = MemoryProviderMetadata(
            name="builtin",
            version="1.0.0",
            description="Built-in memory using telemetry_store + BM25 full-text search",
        )
        self._telemetry_store: Any = None  # TelemetryStore
        self._bm25: Any = None  # BM25Search
        self._orchestrator_home: Path | None = None
        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self, orchestrator_home: Path) -> None:
        """Initialize TelemetryStore and BM25 search connections."""
        from ..telemetry_store import TelemetryStore
        from ..bm25_search import get_bm25_search

        self._orchestrator_home = orchestrator_home
        self._telemetry_store = TelemetryStore()
        self._bm25 = get_bm25_search(
            db_path=str(orchestrator_home / "search.db"),
        )
        self._initialized = True
        logger.debug("BuiltinMemoryProvider initialized")

    async def shutdown(self) -> None:
        """Flush telemetry and close connections."""
        if self._telemetry_store is not None:
            try:
                await self._telemetry_store.close()
            except Exception:
                pass
        if self._bm25 is not None:
            try:
                self._bm25.close()
            except Exception:
                pass
        self._initialized = False
        logger.debug("BuiltinMemoryProvider shut down")

    # ── Data flow ──────────────────────────────────────────────────────────

    async def sync_turn(
        self,
        task_id: str,
        task_type: str,
        result: TaskResult,
    ) -> None:
        """Persist task result as routing event and add to BM25 index."""
        from ..models import TaskType as _TT

        if not self._initialized:
            return

        # 1. Record routing event in telemetry
        if self._telemetry_store is not None:
            try:
                tt = _TT(task_type) if task_type else _TT.CODE_GEN
                await self._telemetry_store.record_routing_event(
                    project_id=result.metadata.get("project_id", ""),
                    task_id=task_id,
                    task_type=tt,
                    result=result,
                )
            except Exception as exc:
                logger.warning("sync_turn telemetry failed: %s", exc)

        # 2. Add to BM25 index for cross-project recall
        if self._bm25 is not None:
            try:
                await self._bm25.add_document(
                    doc_id=task_id,
                    project_id=result.metadata.get("project_id", ""),
                    title=f"Task {task_id} ({task_type})",
                    content=result.output[:5000],  # first 5K chars
                    metadata={
                        "score": result.score,
                        "model": result.model_used.value if result.model_used else "",
                        "cost": result.cost_usd,
                    },
                )
            except Exception as exc:
                logger.warning("sync_turn BM25 index failed: %s", exc)

    async def prefetch(self, query: str, limit: int = 5) -> list[MemoryQueryResult]:
        """Query BM25 index for relevant prior task outputs."""
        if not self._initialized or self._bm25 is None:
            return []

        try:
            results = await self._bm25.bm25_search(query, limit=limit)
            return [
                MemoryQueryResult(
                    content=r.content[:2000],
                    score=float(r.score),
                    source=r.project_id,
                    metadata={"doc_id": r.doc_id, "title": r.title},
                )
                for r in results
            ]
        except Exception as exc:
            logger.warning("prefetch BM25 search failed: %s", exc)
            return []
