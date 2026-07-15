"""
ProjectRunnerCallables and ProjectRunState — M3 decoupling helpers
===================================================================
Replaces the ``host=Orchestrator`` back-reference in ProjectRunner.

Instead of ProjectRunner holding a reference to the full Orchestrator object
and calling 13+ private methods on it, the Orchestrator injects:

  - ProjectRunnerCallables  — all execution callbacks as plain callables
  - ProjectRunState         — the small slice of mutable state that both sides
                              need to share (project_id, architecture_rules,
                              entered flag for BUG-003 connection lifecycle)

This makes ProjectRunner independently testable: pass mock callables, no
Orchestrator needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ProjectRunnerCallables:
    """All Orchestrator methods that ProjectRunner needs, injected as callables.

    Field naming mirrors the private method names on Orchestrator so diffs are
    easy to read; callers write ``self._callables.log_summary(state)`` instead
    of ``self._host._log_summary(state)``.
    """

    topological_sort: Callable[..., Any]
    """Return an ordered list of task IDs respecting dependencies."""

    topological_levels: Callable[..., Any]
    """Return tasks grouped into parallel execution levels."""

    make_state: Callable[..., Any]
    """Construct a fresh ProjectState from description + tasks."""

    determine_final_status: Callable[..., Any]
    """Derive the final ProjectStatus from a completed ProjectState."""

    log_summary: Callable[..., None]
    """Emit a human-readable summary of the project outcome."""

    execute_all: Callable[..., Awaitable[Any]]
    """Run all tasks and return the resulting ProjectState."""

    generate_architecture_rules: Callable[..., Awaitable[str]]
    """Generate architecture/style rules for the project."""

    analyze_completed_project: Callable[..., Awaitable[None]]
    """Optional post-completion analysis callback."""

    client: Any
    """LLM client — needed for assumption-surfacing calls in dry_run.

    Required field; declared before the optional ``*_fn`` callables below so the
    dataclass keeps all non-default fields ahead of defaulted ones.
    """

    warm_start_fn: Callable[..., Awaitable[None]] | None = None
    """Async callable to blend historical profiles before execution (run_job)."""

    flush_telemetry_fn: Callable[..., Awaitable[None]] | None = None
    """Async callable to persist telemetry snapshots after completion (run_job)."""

    constitution_gate: Any = None
    """ConstitutionGate pipeline stage (container.constitution_gate), if wired.

    Exposes ``set_constitution()`` so run_project() can swap in a per-run
    constitution (e.g. --from-speckit) before executing precomposed tasks.
    """


@dataclass
class ProjectRunState:
    """Mutable run-level state shared between Orchestrator and ProjectRunner.

    The Orchestrator writes to these fields via the ProjectRunState reference
    it holds; ProjectRunner reads/writes through its own reference to the same
    object.  Both sides see the same data without a direct object reference.
    """

    project_id: str = ""
    """Active project ID (set at the start of run_project)."""

    architecture_rules: str = ""
    """Architecture rules generated for the current project."""

    entered: bool = False
    """True while the Orchestrator is inside an ``async with`` block.

    Used by ProjectRunner's ``finally`` clause (BUG-003 fix): connections are
    only closed here when ``entered=False``; otherwise ``__aexit__`` owns them.
    """

    results: dict[str, Any] = field(default_factory=dict)
    """Task results dict — shared reference so ProjectRunner can read counts."""
