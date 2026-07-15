"""
RunContext — Per-run mutable state for the Orchestrator
========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Encapsulates all mutable per-run state that was previously stored as
instance attributes on ``Orchestrator`` (engine.py).  Each top-level
entry point (``run_project``, ``run_job``, ``run_project_streaming``)
creates a fresh ``RunContext`` so concurrent calls do not share mutable
state.

Usage:
    ctx = RunContext(project_id="abc", budget=Budget())
    engine._run_ctx = ctx
    # All downstream code reads ctx.results, ctx.budget, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import TaskResult


@dataclass
class RunContext:
    """Per-run mutable state for the Orchestrator.

    One instance is created per ``run_project()`` / ``run_job()`` /
    ``run_project_streaming()`` call.  Fields are plain attributes
    (not properties) so existing code that reads ``self.ctx.attr``
    works without indirection.

    Note:
        This is distinct from ``unattended_guard.RunContext``, which
        holds pre-flight check metadata (daily cap, max-retries, etc.)
        and is used by ``UnattendedGuard.validate()``.  The two serve
        different phases of the execution lifecycle and are intentionally
        separate.
    """

    # ── Project identity ──────────────────────────────────────────────
    project_id: str = ""
    """Unique identifier for the current project run."""

    # ── Results and task state ─────────────────────────────────────────
    results: dict[str, TaskResult] = field(default_factory=dict)
    """Per-task results accumulated during this run."""

    channels: dict[str, Any] = field(default_factory=dict)
    """Task communication channels for this run."""

    # ── Budget & policies (set by run_job) ─────────────────────────────
    budget: Any = None
    """Per-run Budget instance.  Set by ``run_job()`` before delegation."""

    active_policies: Any = None
    """PolicySet active for this run.  Set by ``run_job()``."""

    max_parallel_tasks: int = 3
    """Max parallel tasks for this run.  May be overridden by JobSpec."""

    quality_mode: str = "standard"
    """Quality mode for this run.  Set by ``run_job()``."""

    # ── Per-run flags ──────────────────────────────────────────────────
    analyze_on_complete: bool = False
    """Whether to run post-project analysis on completion."""

    # ── Streaming ──────────────────────────────────────────────────────
    event_bus: Any = None
    """Per-streaming-run event bus.  Set by ``run_project_streaming()``."""

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def task_count(self) -> int:
        """Number of tasks that have results."""
        return len(self.results)

    @property
    def completed_count(self) -> int:
        """Number of COMPLETED tasks."""
        from ..models import TaskStatus

        return sum(
            1 for r in self.results.values() if getattr(r, "status", None) == TaskStatus.COMPLETED
        )

    @property
    def failed_count(self) -> int:
        """Number of FAILED tasks."""
        from ..models import TaskStatus

        return sum(
            1 for r in self.results.values() if getattr(r, "status", None) == TaskStatus.FAILED
        )

    def reset(
        self,
        *,
        project_id: str = "",
        budget: Any = None,
        analyze_on_complete: bool = False,
    ) -> None:
        """Reset all mutable state for a new run.

        This is the lightweight path used by ``run_project()``.
        ``run_job()`` additionally sets ``active_policies``,
        ``max_parallel_tasks``, and ``quality_mode`` after calling this.
        """
        self.project_id = project_id
        self.results.clear()
        self.channels.clear()
        self.analyze_on_complete = analyze_on_complete

        # Budget: use the provided one, or keep the existing reference.
        if budget is not None:
            self.budget = budget

        # Reset streaming-scoped fields
        self.event_bus = None
