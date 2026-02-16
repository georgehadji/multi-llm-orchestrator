"""
Agents — AgentPool meta-controller and TaskChannel inter-task messaging.
========================================================================

AgentPool
    Meta-controller that manages multiple Orchestrator instances and can
    run them in parallel via asyncio.gather(). Useful for:
    - A/B testing different optimization backends or policy sets
    - Ensemble runs where you pick the best result from N agents
    - Load distribution across isolated orchestrator instances

    Usage:
        pool = AgentPool()
        pool.add_agent("pareto", Orchestrator(budget=Budget(max_usd=5.0)))
        pool.add_agent("greedy", Orchestrator(budget=Budget(max_usd=5.0)))
        results = asyncio.run(pool.run_parallel({
            "pareto": spec_a,
            "greedy": spec_b,
        }))
        best = pool.best_result(results)

TaskChannel
    Lightweight asyncio.Queue wrapper for passing messages between tasks
    within a single orchestration run. Allows upstream tasks to share
    structured results with downstream tasks without going through the
    dependency context string.

    Usage:
        ch = orch.get_channel("artifacts")
        await ch.put({"type": "schema", "content": schema_json})
        # Later, in a downstream task handler:
        msgs = ch.peek_all()
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .models import Model, ProjectState, TaskResult
from .policy import ModelProfile, JobSpec

if TYPE_CHECKING:
    from .engine import Orchestrator

logger = logging.getLogger("orchestrator.agents")


# ─────────────────────────────────────────────────────────────────────────────
# TaskChannel
# ─────────────────────────────────────────────────────────────────────────────

class TaskChannel:
    """
    asyncio.Queue wrapper for inter-task messaging within a single run.

    Messages are plain dicts — no schema enforcement. Channels are named and
    obtained via Orchestrator.get_channel(name), which creates them lazily.

    peek_all() is non-destructive: it drains the queue and immediately
    re-enqueues the same messages, so subsequent calls see the same data.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """
        Parameters
        ----------
        maxsize : Maximum number of messages. 0 = unbounded (default).
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, message: dict) -> None:
        """Enqueue a message. Blocks if the queue is full (maxsize > 0)."""
        await self._queue.put(message)

    async def get(self) -> dict:
        """Dequeue the next message. Blocks until a message is available."""
        return await self._queue.get()

    def peek_all(self) -> list[dict]:
        """
        Return all queued messages without consuming them.

        Drains the queue into a list, then re-enqueues all messages in the
        same order. O(N) time and memory.
        """
        items: list[dict] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for item in items:
            self._queue.put_nowait(item)
        return items

    def qsize(self) -> int:
        """Return the current number of messages in the queue."""
        return self._queue.qsize()


# ─────────────────────────────────────────────────────────────────────────────
# AgentPool
# ─────────────────────────────────────────────────────────────────────────────

class AgentPool:
    """
    Meta-controller for multiple Orchestrator instances.

    Agents are registered by name. run_parallel() maps agent names to JobSpecs
    and runs them concurrently via asyncio.gather(return_exceptions=True).
    Exceptions from individual agents are logged but do not cancel other agents.

    merge_telemetry() aggregates live ModelProfile data across all agents:
    - EMA fields (avg_latency_ms, quality_score, avg_cost_usd, trust_factor): averaged
    - Counter fields (call_count, failure_count, validator_fail_count): summed
    - success_rate: re-computed as failure_count / call_count
    """

    def __init__(self) -> None:
        self._agents: dict[str, "Orchestrator"] = {}

    def add_agent(self, name: str, orchestrator: "Orchestrator") -> None:
        """Register a named Orchestrator instance."""
        self._agents[name] = orchestrator
        logger.debug("AgentPool: registered agent %r", name)

    def agents(self) -> dict[str, "Orchestrator"]:
        """Return a read-only copy of the registered agents dict."""
        return dict(self._agents)

    async def run_parallel(
        self,
        assignments: dict[str, "JobSpec"],
    ) -> dict[str, ProjectState]:
        """
        Run each assignment on its named agent concurrently.

        Parameters
        ----------
        assignments : dict mapping agent_name → JobSpec

        Returns
        -------
        dict mapping agent_name → ProjectState for agents that completed.
        Agents that raised exceptions are omitted from the result (exception
        is logged at ERROR level).
        """
        names   = list(assignments.keys())
        coros   = [self._agents[name].run_job(assignments[name]) for name in names]
        results_raw = await asyncio.gather(*coros, return_exceptions=True)

        results: dict[str, ProjectState] = {}
        for name, outcome in zip(names, results_raw):
            if isinstance(outcome, Exception):
                logger.error(
                    "AgentPool: agent %r raised during run_job: %s",
                    name, outcome,
                )
            else:
                results[name] = outcome
        return results

    def best_result(
        self,
        results: dict[str, "ProjectState"],
    ) -> Optional["ProjectState"]:
        """
        Return the ProjectState with the highest mean TaskResult.score.

        Skips agents with no task results. Returns None if results is empty.
        """
        if not results:
            return None

        best_state: Optional[ProjectState] = None
        best_score: float = -1.0

        for state in results.values():
            task_results: dict[str, TaskResult] = getattr(state, "results", {})
            if not task_results:
                continue
            scores = [r.score for r in task_results.values() if r.score is not None]
            if not scores:
                continue
            mean_score = sum(scores) / len(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_state = state

        return best_state

    def merge_telemetry(self) -> dict[Model, ModelProfile]:
        """
        Aggregate ModelProfile data from all registered agents into one dict.

        For each model:
        - EMA fields are averaged across agents that have data
        - Counter fields (call_count, failure_count, validator_fail_count) are summed
        - success_rate is re-derived from the merged failure/call counts

        Returns a fresh dict[Model, ModelProfile] (does not mutate any agent's profiles).
        """
        if not self._agents:
            return {}

        # Collect per-model lists of profiles
        from .models import build_default_profiles
        all_profile_dicts: list[dict[Model, ModelProfile]] = [
            agent._profiles for agent in self._agents.values()
            if hasattr(agent, "_profiles")
        ]
        if not all_profile_dicts:
            return build_default_profiles()

        # Start with the union of all model keys
        all_models: set[Model] = set()
        for pd in all_profile_dicts:
            all_models |= set(pd.keys())

        merged: dict[Model, ModelProfile] = {}
        base_profiles = build_default_profiles()

        for model in all_models:
            contributing = [pd[model] for pd in all_profile_dicts if model in pd]
            if not contributing:
                continue

            # Use the first profile as the structural template (static fields)
            template = contributing[0]

            # Sum counters
            total_calls     = sum(p.call_count          for p in contributing)
            total_failures  = sum(p.failure_count       for p in contributing)
            total_val_fails = sum(p.validator_fail_count for p in contributing)

            # Average EMA fields
            n = len(contributing)
            avg_latency   = sum(p.avg_latency_ms   for p in contributing) / n
            lat_p95       = sum(p.latency_p95_ms   for p in contributing) / n
            quality       = sum(p.quality_score    for p in contributing) / n
            trust         = sum(p.trust_factor     for p in contributing) / n
            avg_cost      = sum(p.avg_cost_usd     for p in contributing) / n

            # Re-derive success_rate
            success_rate = (
                (total_calls - total_failures) / total_calls
                if total_calls > 0 else 1.0
            )

            # Build merged profile
            from dataclasses import replace
            mp = replace(
                template,
                call_count          = total_calls,
                failure_count       = total_failures,
                validator_fail_count= total_val_fails,
                avg_latency_ms      = avg_latency,
                latency_p95_ms      = lat_p95,
                quality_score       = quality,
                trust_factor        = trust,
                avg_cost_usd        = avg_cost,
                success_rate        = success_rate,
            )
            merged[model] = mp

        return merged
