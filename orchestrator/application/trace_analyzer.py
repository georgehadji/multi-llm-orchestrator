"""
TraceAnalyzer — Execution Trace Analysis for Bilevel Autoresearch
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads project execution traces from ``ExecutionArchive``, ``StateManager``,
and telemetry to compute metrics used by the outer-loop self-improvement
pipeline:

- **Repetition**: How often the same (model, revision_context) pair is retried
- **Fixation**: Concentration of task routing on a single model/provider
- **Stagnation**: Flat score curves across consecutive iterations
- **Cost-effectiveness**: Score-per-dollar efficiency

These metrics drive the ``SearchStrategyTuner`` (Level 1.5) and inform
the ``MechanismResearcher`` (Level 2) when generating new search mechanisms.

Usage:
    analyzer = TraceAnalyzer(state_mgr, telemetry_store)
    trace = await analyzer.analyze(project_ids=["proj_abc", "proj_def"])
    print(trace.stagnation_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.bilevel.trace_analyzer")


# ── Output data structures ────────────────────────────────────────────────


@dataclass
class TaskTrace:
    """Metrics for a single task across its revision iterations."""

    task_id: str = ""
    task_type: str = ""
    model: str = ""
    num_iterations: int = 0
    final_score: float = 0.0
    total_cost_usd: float = 0.0
    scores: list[float] = field(default_factory=list)
    revision_contexts: list[str] = field(default_factory=list)


@dataclass
class SearchTrace:
    """Aggregated trace analysis for a set of projects."""

    project_ids: list[str] = field(default_factory=list)
    task_traces: list[TaskTrace] = field(default_factory=list)

    # ── Repetition metrics ───────────────────────────────────────────
    total_retries: int = 0
    duplicate_retries: int = 0  # same (model, context) retried
    repetition_rate: float = 0.0  # duplicate_retries / total_retries

    # ── Fixation metrics ─────────────────────────────────────────────
    dominant_model: str = ""
    dominant_model_share: float = 0.0  # fraction of tasks using dominant model

    # ── Stagnation metrics ───────────────────────────────────────────
    stagnated_task_count: int = 0  # tasks with flat score in last N iterations
    stagnation_score: float = 0.0  # fraction of tasks stagnated

    # ── Cost-effectiveness ────────────────────────────────────────────
    avg_cost_per_point: float = 0.0  # average $ per score point
    total_cost: float = 0.0
    avg_score: float = 0.0


# ── Analyzer ──────────────────────────────────────────────────────────────


class TraceAnalyzer:
    """Analyze execution traces for repetition, fixation, and stagnation.

    Args:
        state_mgr: StateManager or compatible port for loading project state.
        telemetry_store: TelemetryStore for historical model profiles.
    """

    def __init__(
        self,
        state_mgr: Any = None,
        telemetry_store: Any = None,
    ) -> None:
        self._state_mgr = state_mgr
        self._telemetry_store = telemetry_store

    async def analyze(
        self,
        project_ids: list[str],
    ) -> SearchTrace:
        """Run full trace analysis on the given project IDs.

        Args:
            project_ids: List of project IDs to analyze.

        Returns:
            A ``SearchTrace`` with aggregated metrics.
        """
        trace = SearchTrace(project_ids=list(project_ids))
        task_traces: list[TaskTrace] = []

        for pid in project_ids:
            project_traces = await self._analyze_project(pid)
            task_traces.extend(project_traces)

        trace.task_traces = task_traces

        # Compute aggregated metrics
        self._compute_repetition(trace)
        self._compute_fixation(trace)
        self._compute_stagnation(trace)
        self._compute_cost_effectiveness(trace)

        return trace

    async def _analyze_project(
        self,
        project_id: str,
    ) -> list[TaskTrace]:
        """Analyze a single project's execution traces.

        Loads project state from StateManager, extracts per-task
        revision history, and builds ``TaskTrace`` objects.

        Returns:
            List of ``TaskTrace``, one per task.
        """
        task_traces: list[TaskTrace] = []

        if self._state_mgr is None:
            logger.debug("No state_mgr available — returning empty trace for %s", project_id)
            return task_traces

        try:
            state = await self._state_mgr.load_project(project_id)
        except Exception as exc:
            logger.warning("Failed to load project %s: %s", project_id, exc)
            return task_traces

        if state is None:
            return task_traces

        # Iterate over task results
        results = getattr(state, "results", {}) or {}
        for task_id, result in results.items():
            task_trace = TaskTrace(
                task_id=task_id,
                task_type=str(getattr(result, "task_type", "")),
                model=str(getattr(result, "model", "")),
                num_iterations=getattr(result, "iterations", 1),
                final_score=getattr(result, "score", 0.0),
                total_cost_usd=getattr(result, "cost_usd", 0.0),
                scores=list(getattr(result, "score_history", []) or []),
                revision_contexts=list(getattr(result, "revision_contexts", []) or []),
            )
            task_traces.append(task_trace)

        return task_traces

    # ── Metric computation ────────────────────────────────────────────

    def _compute_repetition(self, trace: SearchTrace) -> None:
        """Compute repetition rate: fraction of retries that duplicate a prior attempt."""
        total = 0
        duplicates = 0
        seen_pairs: set[tuple[str, str]] = set()

        for tt in trace.task_traces:
            for ctx in tt.revision_contexts:
                pair = (tt.model, ctx[:100])  # truncate to avoid noise
                total += 1
                if pair in seen_pairs:
                    duplicates += 1
                else:
                    seen_pairs.add(pair)

        trace.total_retries = total
        trace.duplicate_retries = duplicates
        trace.repetition_rate = duplicates / total if total > 0 else 0.0

    def _compute_fixation(self, trace: SearchTrace) -> None:
        """Compute model fixation: concentration on a single model."""
        model_counts: dict[str, int] = {}
        for tt in trace.task_traces:
            model_counts[tt.model] = model_counts.get(tt.model, 0) + 1

        if not model_counts:
            return

        dominant = max(model_counts, key=model_counts.get)
        trace.dominant_model = dominant
        trace.dominant_model_share = model_counts[dominant] / len(trace.task_traces)

    def _compute_stagnation(self, trace: SearchTrace, window: int = 3) -> None:
        """Compute stagnation: tasks whose last ``window`` scores show no improvement.

        A task is considered stagnated when its last ``window`` scores are
        all within 1% of each other (flat tail).
        """
        stagnated = 0
        for tt in trace.task_traces:
            if len(tt.scores) < window:
                continue
            tail = tt.scores[-window:]
            if max(tail) - min(tail) < 0.01 * max(tail) if max(tail) > 0 else True:
                stagnated += 1

        trace.stagnated_task_count = stagnated
        trace.stagnation_score = stagnated / len(trace.task_traces) if trace.task_traces else 0.0

    def _compute_cost_effectiveness(self, trace: SearchTrace) -> None:
        """Compute average cost per score point and overall metrics."""
        total_cost = sum(tt.total_cost_usd for tt in trace.task_traces)
        total_score = sum(tt.final_score for tt in trace.task_traces)
        task_count = len(trace.task_traces)

        trace.total_cost = total_cost
        trace.avg_score = total_score / task_count if task_count > 0 else 0.0
        trace.avg_cost_per_point = total_cost / total_score if total_score > 0 else 0.0
