"""
SkillManager — Facade over all per-TaskType SkillOptimizers
===========================================================
Owns the trajectory buffer and epoch-triggering logic for every TaskType.

Responsibilities
----------------
- ``record_trajectory(t)`` — persist to SkillStore, buffer in memory, fire
  epoch when the buffer reaches epoch_size.
- ``best_skill(task_type)`` — return the current best skill doc (or None) for
  injection into the worker system prompt.
- Fire-and-forget epoch runs via ``asyncio.create_task`` so they never block
  the hot task-execution path.
- Swallow epoch failures and log — a broken optimizer must never crash the main loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from ..models import TaskType
from ..models_skill import Trajectory
from .skill_optimizer import SkillOptimizer

if TYPE_CHECKING:
    from ..domain.ports import LLMClient, SkillStorePort

logger = logging.getLogger("orchestrator.skill_manager")


class SkillManager:
    """Manages one :class:`SkillOptimizer` per :class:`TaskType`.

    Instantiate in ``Orchestrator.__init__()`` when
    ``flags.skill_optimization_enabled`` is True; call ``await close()`` in
    ``Orchestrator.close()`` / ``__aexit__``.
    """

    def __init__(
        self,
        optimizer_client: "LLMClient",
        skill_store: "SkillStorePort",
        epoch_size: int = 10,
        edit_budget: int = 150,
        validation_fraction: float = 0.2,
        min_trajectories: int = 5,
        slow_update_every: int = 5,
        enabled: bool = True,
    ) -> None:
        self._client = optimizer_client
        self._store = skill_store
        self._epoch_size = epoch_size
        self._enabled = enabled

        # One optimizer per TaskType (lazy creation)
        self._optimizers: dict[TaskType, SkillOptimizer] = {
            tt: SkillOptimizer(
                task_type=tt,
                optimizer_client=optimizer_client,
                skill_store=skill_store,
                edit_budget=edit_budget,
                validation_fraction=validation_fraction,
                min_trajectories=min_trajectories,
                slow_update_every=slow_update_every,
            )
            for tt in TaskType
        }

        # In-memory trajectory buffers — flushed to epoch when full
        self._buffers: dict[TaskType, list[Trajectory]] = defaultdict(list)

        # Track running epoch tasks so we can await them on close
        self._running_epochs: set[asyncio.Task] = set()  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_trajectory(self, t: Trajectory) -> None:
        """Persist trajectory and fire an epoch when the buffer is full."""
        if not self._enabled:
            return
        try:
            await self._store.save_trajectory(t)
        except Exception as exc:
            logger.warning("SkillManager: failed to save trajectory: %s", exc)
            return

        self._buffers[t.task_type].append(t)

        if len(self._buffers[t.task_type]) >= self._epoch_size:
            batch = list(self._buffers[t.task_type])
            self._buffers[t.task_type].clear()
            task = asyncio.create_task(self._run_epoch_safe(t.task_type, batch))
            self._running_epochs.add(task)
            task.add_done_callback(self._running_epochs.discard)

    async def best_skill(self, task_type: TaskType) -> str | None:
        """Return the current best skill document for *task_type*, or ``None``."""
        if not self._enabled:
            return None
        try:
            return await self._optimizers[task_type].best_skill()
        except Exception as exc:
            logger.warning("SkillManager: best_skill(%s) failed: %s", task_type.value, exc)
            return None

    async def wait_for_epochs(self) -> None:
        """Await any in-flight epoch tasks without closing the store.

        Use this as a sync barrier when the caller still needs the store
        afterwards (e.g. to read the persisted best skill). ``close()`` owns
        the store's lifecycle and must not be used merely to flush epochs.
        """
        if self._running_epochs:
            logger.debug("SkillManager: waiting for %d epoch tasks", len(self._running_epochs))
            await asyncio.gather(*self._running_epochs, return_exceptions=True)

    async def close(self) -> None:
        """Wait for any in-flight epoch tasks and close the store."""
        await self.wait_for_epochs()
        await self._store.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_epoch_safe(self, task_type: TaskType, trajectories: list[Trajectory]) -> None:
        """Run one optimizer epoch; swallow all exceptions."""
        try:
            result = await self._optimizers[task_type].run_epoch(trajectories)
            if result.accepted:
                logger.info(
                    "SkillManager: epoch %d for %s accepted — score %.4f → %.4f",
                    result.epoch,
                    task_type.value,
                    result.score_before,
                    result.score_after,
                )
            else:
                logger.debug(
                    "SkillManager: epoch %d for %s rejected — %s",
                    result.epoch,
                    task_type.value,
                    result.rejection_reason,
                )
        except Exception as exc:
            logger.exception("SkillManager: epoch for %s crashed: %s", task_type.value, exc)
