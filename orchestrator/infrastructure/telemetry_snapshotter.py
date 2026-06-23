"""
TelemetrySnapshotter — Fire-and-forget telemetry persistence + background task management
========================================================================================
Extracted from engine.py (Cluster 1 of the Strangler Fig extraction).

Owns:
- Flushing ModelProfile snapshots to TelemetryStore
- Tracking and cleanup of background asyncio tasks
- Periodic cleanup timer
- Recording routing events to TelemetryStore
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models import TaskType

logger = logging.getLogger(__name__)


class TelemetrySnapshotter:
    """
    Fire-and-forget telemetry persistence with background task lifecycle.

    Usage:
        snapshotter = TelemetrySnapshotter(
            telemetry_store=container.telemetry_store,
            get_active_profiles_fn=lambda: [...],
            background_tasks=engine._background_tasks,
        )
        await snapshotter.flush_snapshots(project_id)
        snapshotter.start_periodic_cleanup(interval_seconds=300)
    """

    def __init__(
        self,
        telemetry_store: Any,
        get_active_profiles_fn: Callable[[], list],
        background_tasks: set[asyncio.Task] | None = None,
    ):
        self._telemetry_store = telemetry_store
        self._get_active_profiles = get_active_profiles_fn
        self._background_tasks: set[asyncio.Task] = background_tasks or set()
        self._cleanup_timer: asyncio.Task | None = None

    # ── Snapshot flushing ──────────────────────────────────────────────────

    async def flush_snapshots(self, project_id: str) -> None:
        """
        Fire-and-forget: snapshot each ModelProfile that was used this run.
        Only profiles with call_count >= 1 are written.
        Uses asyncio.create_task so the hot path is never blocked.
        """

        async def _write_snapshots() -> None:
            active_profiles = self._get_active_profiles()
            if active_profiles:
                try:
                    result = await self._telemetry_store.record_snapshots_batch(
                        project_id, active_profiles
                    )
                    if isinstance(result, dict):
                        if result.get("failed", 0) > 0:
                            logger.warning(
                                f"Telemetry batch: {result['success']} succeeded, "
                                f"{result['failed']} failed"
                            )
                            for err in result.get("errors", [])[:5]:
                                logger.warning(
                                    f"  - {err.get('model', 'unknown')}: {err.get('error', 'unknown')}"
                                )
                        else:
                            logger.debug(
                                f"Telemetry batch flush complete for {len(active_profiles)} models"
                            )
                except Exception as exc:
                    logger.warning(f"TelemetryStore.record_snapshots_batch failed: {exc}")
            else:
                logger.debug("No active profiles to flush")

        task = asyncio.create_task(_write_snapshots())
        self._background_tasks.add(task)
        task.add_done_callback(self._cleanup_task_callback)

    # ── Background task lifecycle ──────────────────────────────────────────

    def _cleanup_task_callback(self, task: asyncio.Task) -> None:
        """Done-callback: remove task from the strong-reference set and log failures."""
        self._background_tasks.discard(task)
        if task.cancelled():
            logger.debug("Background task was cancelled")
        elif task.exception() is not None:
            logger.warning("Background task failed: %s", task.exception())
        else:
            logger.debug("Background task completed successfully")

    async def cleanup_done_tasks(self) -> int:
        """Remove completed tasks from the tracking set. Returns number removed."""
        if not self._background_tasks:
            return 0
        done = {t for t in self._background_tasks if t.done()}
        self._background_tasks -= done
        logger.debug(
            "Background tasks cleaned up: %d done, %d still running",
            len(done),
            len(self._background_tasks),
        )
        return len(done)

    def start_periodic_cleanup(self, interval_seconds: int = 300) -> None:
        """Start periodic cleanup timer for completed background tasks."""

        async def _cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_done_tasks()

        self._cleanup_timer = asyncio.create_task(_cleanup_loop())
        logger.info("Started periodic cleanup timer (interval=%ds)", interval_seconds)

    def stop_periodic_cleanup(self) -> None:
        """Cancel the periodic cleanup timer if running."""
        if self._cleanup_timer is not None and not self._cleanup_timer.done():
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

    # ── Routing events ─────────────────────────────────────────────────────

    async def record_routing_event(
        self,
        project_id: str,
        task_id: str,
        task_type: TaskType,
        result: Any,
    ) -> None:
        """Fire-and-forget wrapper: record a routing event, swallowing exceptions."""
        try:
            await self._telemetry_store.record_routing_event(project_id, task_id, task_type, result)
        except Exception as exc:
            logger.warning("TelemetryStore.record_routing_event failed: %s", exc)
