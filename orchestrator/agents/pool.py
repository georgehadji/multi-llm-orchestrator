"""
AgentPool meta-controller and TaskChannel inter-task messaging.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Moved here from ``orchestrator/agents.py``, which was shadowed by this package and
therefore unimportable by construction (P3-SHADOW1) — ``engine.py:158`` has been
doing ``from .agents import TaskChannel`` and silently binding ``None`` via its
``except ImportError`` fallback for the lifetime of the file.

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
        ch = TaskChannel()
        await ch.put({"type": "schema", "content": schema_json})
        # Later, in a downstream task handler:
        msgs = ch.peek_all()

    NOTE: this class previously documented ``orch.get_channel(name)`` as the way
    to obtain a channel. No such method exists on Orchestrator and none ever has
    (verified repo-wide), so callers must construct channels directly. Adding a
    channel registry is new behaviour and belongs in a service module rather than
    in engine.py, per this repo's Mediator rule.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import Orchestrator
    from ..models import Model, ProjectState, TaskResult
    from ..policy import JobSpec, ModelProfile

logger = logging.getLogger("orchestrator.agents.pool")


# ─────────────────────────────────────────────────────────────────────────────
# TaskChannel
# ─────────────────────────────────────────────────────────────────────────────


class TaskChannel:
    """
    asyncio.Queue wrapper for inter-task messaging within a single run.

    Messages are plain dicts — no schema enforcement.

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

        Not atomic: a concurrent put() between the drain and the re-enqueue
        would be ordered after the peeked messages rather than before them.
        Callers needing a consistent snapshot under concurrency should hold
        their own lock.
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


def _agent_profiles(agent: Orchestrator) -> dict | None:
    """Return an agent's live ModelProfile dict, or None if it exposes none.

    The profiles live at ``orchestrator._c.planner._profiles`` (see
    ``engine.py:788``). The original code here read ``agent._profiles``, an
    attribute Orchestrator does not define, so its ``hasattr`` guard was always
    False and merge_telemetry() could only ever return default profiles — a
    70-line aggregation that never aggregated anything.
    """
    container = getattr(agent, "_c", None)
    planner = getattr(container, "planner", None)
    profiles = getattr(planner, "_profiles", None)
    if profiles:
        return dict(profiles)
    # Objects that expose profiles directly (test doubles, future refactors).
    direct = getattr(agent, "_profiles", None)
    return dict(direct) if direct else None


class AgentPool:
    """
    Meta-controller for multiple Orchestrator instances.

    Agents are registered by name. run_parallel() maps agent names to JobSpecs
    and runs them concurrently via asyncio.gather(return_exceptions=True).
    Exceptions from individual agents are logged but do not cancel other agents.

    merge_telemetry() aggregates live ModelProfile data across all agents:
    - EMA fields (avg_latency_ms, quality_score, avg_cost_usd, trust_factor):
      averaged, weighted by call_count
    - Counter fields (call_count, failure_count, validator_fail_count): summed
    - success_rate: re-computed from the merged counters
    """

    def __init__(self) -> None:
        self._agents: dict[str, Orchestrator] = {}

    def add_agent(self, name: str, orchestrator: Orchestrator) -> None:
        """Register a named Orchestrator instance."""
        self._agents[name] = orchestrator
        logger.debug("AgentPool: registered agent %r", name)

    def agents(self) -> dict[str, Orchestrator]:
        """Return a read-only copy of the registered agents dict."""
        return dict(self._agents)

    async def run_parallel(
        self,
        assignments: dict[str, JobSpec],
    ) -> dict[str, ProjectState]:
        """
        Run each assignment on its named agent concurrently.

        Parameters
        ----------
        assignments : dict mapping agent_name → JobSpec

        Returns
        -------
        dict mapping agent_name → ProjectState for agents that completed.
        Agents that raised are omitted from the result (logged at ERROR level),
        as are assignments naming an agent that was never registered.
        """
        # An unregistered name used to raise KeyError while *building* the coros,
        # so a single bad assignment prevented every other agent from running at
        # all — the opposite of this method's documented isolation guarantee.
        names = []
        for name in assignments:
            if name in self._agents:
                names.append(name)
            else:
                logger.error("AgentPool: no agent registered as %r — skipping its assignment", name)
        if not names:
            return {}

        coros = [self._agents[name].run_job(assignments[name]) for name in names]
        results_raw = await asyncio.gather(*coros, return_exceptions=True)

        results: dict[str, ProjectState] = {}
        for name, outcome in zip(names, results_raw, strict=False):
            # BaseException, not Exception: gather(return_exceptions=True) also
            # returns CancelledError, which is a BaseException and would
            # otherwise be recorded as a successful ProjectState.
            if isinstance(outcome, BaseException):
                logger.error(
                    "AgentPool: agent %r raised during run_job: %s",
                    name,
                    outcome,
                )
            else:
                results[name] = outcome
        return results

    def best_result(
        self,
        results: dict[str, ProjectState],
    ) -> ProjectState | None:
        """
        Return the ProjectState with the highest mean TaskResult.score.

        Skips agents with no task results. Returns None if results is empty.
        """
        if not results:
            return None

        best_state: ProjectState | None = None
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
        - EMA fields are averaged, weighted by call_count
        - Counter fields are summed
        - success_rate is re-derived from the merged failure/call counts

        Returns a fresh dict (does not mutate any agent's profiles). Falls back
        to default profiles when no agent exposes live ones.
        """
        from ..application.model_profile_builder import build_default_profiles

        if not self._agents:
            return {}

        all_profile_dicts: list[dict[Model, ModelProfile]] = [
            profiles
            for profiles in (_agent_profiles(agent) for agent in self._agents.values())
            if profiles
        ]
        if not all_profile_dicts:
            logger.warning(
                "AgentPool.merge_telemetry: none of the %d registered agents exposed live "
                "model profiles — returning defaults, so the merge is a no-op",
                len(self._agents),
            )
            return build_default_profiles()

        # Start with the union of all model keys
        all_models: set[Model] = set()
        for pd in all_profile_dicts:
            all_models |= set(pd.keys())

        merged: dict[Model, ModelProfile] = {}

        for model in all_models:
            contributing = [pd[model] for pd in all_profile_dicts if model in pd]
            if not contributing:
                continue

            # Use the first profile as the structural template (static fields)
            template = contributing[0]

            # Sum counters
            total_calls = sum(p.call_count for p in contributing)
            total_failures = sum(p.failure_count for p in contributing)
            total_val_fails = sum(p.validator_fail_count for p in contributing)

            # Weighted average for EMA fields — weight by call_count so agents
            # that have made more calls contribute proportionally more to the
            # merged estimate. Falls back to simple average if all counts are 0.
            weight_sum = total_calls or len(contributing)  # avoid division by zero

            def _wavg(
                attr: str, _c: list = contributing, _t: int = total_calls, _w: int = weight_sum
            ) -> float:
                if _t == 0:
                    return sum(getattr(p, attr) for p in _c) / len(_c)
                return sum(getattr(p, attr) * p.call_count for p in _c) / _w

            avg_latency = _wavg("avg_latency_ms")
            lat_p95 = _wavg("latency_p95_ms")
            quality = _wavg("quality_score")
            trust = _wavg("trust_factor")
            avg_cost = _wavg("avg_cost_usd")

            # Re-derive success_rate
            success_rate = (total_calls - total_failures) / total_calls if total_calls > 0 else 1.0

            # Build merged profile
            from dataclasses import replace

            mp = replace(
                template,
                call_count=total_calls,
                failure_count=total_failures,
                validator_fail_count=total_val_fails,
                avg_latency_ms=avg_latency,
                latency_p95_ms=lat_p95,
                quality_score=quality,
                trust_factor=trust,
                avg_cost_usd=avg_cost,
                success_rate=success_rate,
            )
            merged[model] = mp

        return merged
