"""
Memory Manager — Orchestrates Memory Providers + Periodic Consolidation
=========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Orchestrates multiple MemoryProvider implementations. Wires data flow:
- After each task completion: sync_turn() to all providers
- Before each task: prefetch_all() for relevant context
- After N projects: maybe_consolidate() extracts cross-project insights

Integration: Called from engine.py.run_project() and engine.py._execute_task().
The MemoryManager is injected via Orchestrator.__init__().

Usage:
    manager = MemoryManager()
    await manager.sync_turn(task_id, task_type, result)
    context = await manager.prefetch_all("user query or prompt")
    await manager.maybe_consolidate()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..plugins.memory_provider import (
    BuiltinMemoryProvider,
    MemoryProvider,
)

if TYPE_CHECKING:
    from ..models import TaskResult

logger = logging.getLogger("orchestrator.memory")


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────


class MemoryManager:
    """Orchestrates memory providers and periodic consolidation.

    Lifecycle (called from Orchestrator):
        1. manager.initialize(orchestrator_home, providers) — during Orchestrator.__init__
        2. manager.prefetch_all(query) — before each task generation
        3. manager.sync_turn(task_id, task_type, result) — after each task
        4. manager.maybe_consolidate() — after project completion
        5. manager.shutdown() — on Orchestrator shutdown
    """

    def __init__(
        self,
        providers: list[MemoryProvider] | None = None,
        nudge_interval: int = 5,
    ) -> None:
        """Initialize manager.

        Args:
            providers: List of MemoryProvider instances. Defaults to
                [BuiltinMemoryProvider()] when None.
            nudge_interval: Number of projects between automatic
                consolidation runs. Default 5.
        """
        self._providers: list[MemoryProvider] = providers or [BuiltinMemoryProvider()]
        self._nudge_interval: int = nudge_interval
        self._projects_since_consolidation: int = 0
        self._initialized: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self, orchestrator_home: Path) -> None:
        """Initialize all registered memory providers.

        Args:
            orchestrator_home: Path to the orchestrator cache directory
                (typically ~/.orchestrator_cache/).
        """
        results = await asyncio.gather(
            *(p.initialize(orchestrator_home) for p in self._providers),
            return_exceptions=True,
        )
        for provider, result in zip(self._providers, results):
            if isinstance(result, Exception):
                logger.error(
                    "MemoryProvider '%s' failed to initialize: %s",
                    provider.metadata.name,
                    result,
                )
        self._initialized = True
        logger.info(
            "MemoryManager initialized with %d provider(s): %s",
            len(self._providers),
            [p.metadata.name for p in self._providers],
        )

    async def shutdown(self) -> None:
        """Shutdown all providers gracefully."""
        if not self._initialized:
            return
        results = await asyncio.gather(
            *(p.shutdown() for p in self._providers),
            return_exceptions=True,
        )
        for provider, result in zip(self._providers, results):
            if isinstance(result, Exception):
                logger.warning(
                    "MemoryProvider '%s' shutdown error: %s",
                    provider.metadata.name,
                    result,
                )
        self._initialized = False
        logger.debug("MemoryManager shut down")

    # ── Data flow ──────────────────────────────────────────────────────────

    async def sync_turn(
        self,
        task_id: str,
        task_type: str,
        result: TaskResult,
    ) -> None:
        """Sync a completed task to all memory providers.

        Fire-and-forget: exceptions from individual providers are logged
        and do not block other providers.

        Args:
            task_id: Unique task identifier.
            task_type: String task type (e.g. ``"code_generation"``).
            result: The completed TaskResult.
        """
        if not self._initialized:
            return

        tasks = [p.sync_turn(task_id, task_type, result) for p in self._providers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def prefetch_all(self, query: str, limit: int = 3) -> str:
        """Aggregate relevant context from all memory providers.

        Args:
            query: The task prompt or project description to match against.
            limit: Maximum results per provider.

        Returns:
            Concatenated context string from all providers, or empty string
            when no context is available.
        """
        if not self._initialized:
            return ""

        results = await asyncio.gather(
            *(p.prefetch(query, limit=limit) for p in self._providers),
            return_exceptions=True,
        )

        parts: list[str] = []
        for provider, res in zip(self._providers, results):
            if isinstance(res, Exception):
                logger.debug(
                    "MemoryProvider '%s' prefetch error: %s",
                    provider.metadata.name,
                    res,
                )
                continue
            if isinstance(res, list):
                for r in res:
                    if r.content:
                        parts.append(r.content)

        if not parts:
            return ""

        return "\n\n---\n\n".join(parts)

    async def maybe_consolidate(self) -> None:
        """Trigger cross-project consolidation after N projects.

        Called after each project completes. When ``projects_since_consolidation``
        reaches ``nudge_interval``, fires the consolidation loop.
        """
        if not self._initialized:
            return

        self._projects_since_consolidation += 1
        if self._projects_since_consolidation >= self._nudge_interval:
            logger.info(
                "MemoryManager consolidation triggered after %d projects",
                self._projects_since_consolidation,
            )
            self._projects_since_consolidation = 0
            # Delegates to ConsolidationLoop from Phase 3.4
            try:
                from .consolidation import ConsolidationLoop

                loop = ConsolidationLoop()
                await loop.run()
            except Exception as exc:
                logger.warning("Consolidation failed: %s", exc)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
