"""
PipelineRunner — Core execution pipeline delegation.
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the ENGINE_OPTIMIZATION_PLAN Phase 6: extracts pipeline execution
methods from Orchestrator while keeping _execute_task on the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Task, TaskType

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Executes the generate->critique->revise->evaluate pipeline.

    Holds a back-reference to the Orchestrator for access to state,
    budget, telemetry, and _execute_task.  All public methods mirror
    the corresponding Orchestrator methods.
    """

    def __init__(self, orchestrator: object) -> None:
        self._orch = orchestrator

    # ── Context methods (Phase 6-A) ────────────────────────────────────

    def build_system_prompt(self, task_type: str = "") -> str:
        """Build system prompt based on current quality_mode."""
        orch = self._orch
        profiles = getattr(orch, "_profiles", {})
        mode = "standard"
        for p in profiles.values():
            if hasattr(p, "quality_mode"):
                mode = p.quality_mode
                break
        # Minimal system prompt — full implementation in prompt_builder.py
        if mode == "production":
            return "You are a senior software engineer. Write production-quality code."
        return "You are a software engineer. Write clean, working code."

    def build_project_context(self) -> str:
        """Build project context from existing results."""
        orch = self._orch
        results = getattr(orch, "results", {})
        context_parts = []
        for task_id, result in results.items():
            if hasattr(result, "output") and result.output:
                context_parts.append(f"## {task_id}\n```python\n{result.output[:2000]}\n```")
        if context_parts:
            return "## Existing Code Context\n\n" + "\n\n".join(context_parts)
        return ""

    # ── Cache warm (Phase 6-B) ─────────────────────────────────────────

    async def warm_cache_for_level(self, tasks: dict, runnable: list) -> None:
        """OPTIMIZATION: Proactively warm cache before parallel execution."""
        orch = self._orch
        cache = getattr(orch, "cache", None)
        if cache is None:
            return
        for tid in runnable:
            task = tasks.get(tid)
            if task and hasattr(task, "prompt"):
                try:
                    await cache.get(task.prompt)
                except Exception:
                    pass  # cache warm failures are non-critical

    # ── Execute all (Phase 6-C) ────────────────────────────────────────

    async def execute_all(self, tasks: dict, execution_order: list, policy=None) -> None:
        """Execute all tasks respecting dependencies.

        Uses topological levels for parallel execution within each level.
        Core _execute_task stays on Orchestrator — PipelineRunner manages
        the execution ordering, concurrency, and telemetry.
        """
        orch = self._orch
        guard = getattr(orch, "_task_guard", None)
        telemetry = getattr(orch, "_telemetry", None)

        levels = self._topological_levels(execution_order, tasks)

        for level_idx, level in enumerate(levels):
            # Warm cache before parallel execution
            await self.warm_cache_for_level(tasks, level)

            if guard:
                async with guard:
                    coros = [orch._execute_task(tasks[tid], policy) for tid in level]
                    results = await asyncio.gather(*coros, return_exceptions=True)
            else:
                coros = [orch._execute_task(tasks[tid], policy) for tid in level]
                results = await asyncio.gather(*coros, return_exceptions=True)

            # Store results and record telemetry
            for tid, r in zip(level, results):
                if isinstance(r, Exception):
                    logger.error(f"Task {tid} failed: {r}")
                    continue
                orch.results[tid] = r
                if telemetry:
                    try:
                        telemetry.record_call(
                            model=getattr(r, "model", None),
                            latency_ms=getattr(r, "latency_ms", 0),
                            cost_usd=getattr(r, "cost_usd", 0),
                            success=getattr(r, "score", 0) > 0,
                            quality_score=getattr(r, "score", None),
                        )
                    except Exception:
                        pass

    @staticmethod
    def _topological_levels(execution_order: list, tasks: dict) -> list[list]:
        """Group tasks into execution levels for parallel processing."""
        levels = []
        remaining = set(execution_order)
        while remaining:
            level = []
            for tid in list(remaining):
                task = tasks.get(tid)
                deps = getattr(task, "dependencies", [])
                if all(d not in remaining for d in deps):
                    level.append(tid)
            for tid in level:
                remaining.discard(tid)
            if level:
                levels.append(level)
        return levels

    async def execute_all_with_retry(self, tasks, execution_order, policy=None):
        """PHASE D4: Execute with auto-retry on better models (Undo+Retry pattern).

        When a task fails, retries it once with a higher-tier model before
        marking it as failed. Uses the existing RetryTemplate from resilience.py.
        """
        orch = self._orch
        guard = getattr(orch, "_task_guard", None)

        levels = self._topological_levels(execution_order, tasks)

        for level_idx, level in enumerate(levels):
            await self.warm_cache_for_level(tasks, level)

            if guard:
                async with guard:
                    coros = [self._execute_with_retry(tasks[tid], policy) for tid in level]
                    results = await asyncio.gather(*coros, return_exceptions=True)
            else:
                coros = [self._execute_with_retry(tasks[tid], policy) for tid in level]
                results = await asyncio.gather(*coros, return_exceptions=True)

            for tid, r in zip(level, results):
                if isinstance(r, Exception):
                    logger.error(f"Task {tid} failed after retry: {r}")
                    continue
                orch.results[tid] = r

    async def _execute_with_retry(self, task, policy=None):
        """Execute a task with one retry on a higher-tier model if it fails."""
        orch = self._orch

        # First attempt
        result = await orch._execute_task(task, policy)
        if getattr(result, "score", 0) >= 0.5:
            return result

        # Try retry on a better model
        from .resilience import RetryTemplate

        retry = RetryTemplate(max_attempts=1, base_delay=0.5)
        for attempt in retry:
            try:
                with attempt:
                    # Escalate tier before retry
                    if hasattr(orch, "_escalate_tier") and hasattr(task, "type"):
                        orch._escalate_tier(task.type)
                    result = await orch._execute_task(task, policy)
                    if getattr(result, "score", 0) > 0:
                        logger.info(
                            f"Retry success for {getattr(task, 'id', '?')}: "
                            f"score={getattr(result, 'score', 0):.3f}"
                        )
                        return result
            except Exception as exc:
                logger.warning(f"Retry attempt failed for {getattr(task, 'id', '?')}: {exc}")

        return result  # Return original (failed) result
