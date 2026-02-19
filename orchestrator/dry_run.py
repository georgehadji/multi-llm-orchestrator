"""
Dry-run / Execution Plan — Improvement 12
==========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Shows what the orchestrator *would* do without making any execution API calls.

The dry_run() method:
  1. Calls _decompose() to get the task list (one real API call to decompose)
  2. Groups tasks into parallel levels via _topological_levels()
  3. Estimates per-task cost from the static COST_TABLE
  4. Returns an ExecutionPlan — no task execution happens

CLI usage:
    python -m orchestrator --project "..." --criteria "..." --dry-run
    python -m orchestrator --file spec.yaml --dry-run

Programmatic usage:
    from orchestrator import Orchestrator
    plan = asyncio.run(orch.dry_run("build FastAPI auth", "tests pass"))
    print(plan.render())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Task


# ─── Token estimates for cost prediction ─────────────────────────────────────
# Average prompt + output tokens per task type (rough heuristic)
_TOKEN_ESTIMATES: dict[str, tuple[int, int]] = {
    "code_generation":  (2_500, 1_500),
    "code_review":      (2_000, 800),
    "complex_reasoning": (1_500, 1_000),
    "creative_writing":  (1_200, 1_000),
    "data_extraction":  (2_000, 600),
    "summarization":    (3_000, 500),
    "evaluation":       (1_500, 400),
}
_DEFAULT_TOKENS = (1_500, 600)


@dataclass
class TaskPlan:
    """Plan for a single task in the execution plan."""
    task_id: str
    task_type: str
    prompt_preview: str       # first 80 chars of prompt
    dependencies: list[str]
    parallel_level: int       # 0-indexed level (tasks in the same level run in parallel)
    primary_model: str        # model that will be used
    estimated_cost_usd: float
    acceptance_threshold: float
    max_iterations: int


@dataclass
class ExecutionPlan:
    """
    Complete dry-run plan for a project.

    Attributes
    ----------
    project_description:  The original project description.
    success_criteria:     The original success criteria.
    tasks:                Ordered list of TaskPlan (topological order).
    parallel_levels:      Tasks grouped into parallel execution levels.
    estimated_total_cost: Sum of per-task cost estimates.
    num_parallel_levels:  Number of sequential execution waves.
    """
    project_description: str
    success_criteria: str
    tasks: list[TaskPlan] = field(default_factory=list)
    parallel_levels: list[list[str]] = field(default_factory=list)
    estimated_total_cost: float = 0.0
    num_parallel_levels: int = 0

    def render(self) -> str:
        """Return a human-readable text representation of the execution plan."""
        return DryRunRenderer.render(self)


class DryRunRenderer:
    """Renders an ExecutionPlan as human-readable text."""

    @staticmethod
    def render(plan: "ExecutionPlan") -> str:
        lines: list[str] = []
        lines.append("=" * 64)
        lines.append("DRY-RUN: Execution Plan")
        lines.append("=" * 64)
        lines.append(f"Project : {plan.project_description[:80]}")
        lines.append(f"Criteria: {plan.success_criteria[:80]}")
        lines.append(
            f"Tasks   : {len(plan.tasks)} task(s) across "
            f"{plan.num_parallel_levels} parallel level(s)"
        )
        lines.append(f"Est. cost: ${plan.estimated_total_cost:.4f}")
        lines.append("")

        for level_idx, level_task_ids in enumerate(plan.parallel_levels):
            parallel_note = " (in parallel)" if len(level_task_ids) > 1 else ""
            lines.append(f"── Level {level_idx}{parallel_note} ──")
            for tid in level_task_ids:
                tp = next((t for t in plan.tasks if t.task_id == tid), None)
                if tp is None:
                    continue
                dep_str = (
                    f"  deps=[{', '.join(tp.dependencies)}]"
                    if tp.dependencies else ""
                )
                lines.append(
                    f"  {tp.task_id}  [{tp.task_type}]"
                    f"  model={tp.primary_model}"
                    f"  est=${tp.estimated_cost_usd:.4f}"
                    f"  threshold={tp.acceptance_threshold}"
                    f"  iters={tp.max_iterations}"
                    f"{dep_str}"
                )
                lines.append(f"    prompt: {tp.prompt_preview!r}")

        lines.append("")
        lines.append("=" * 64)
        lines.append("(No tasks were executed — this is a dry run)")
        return "\n".join(lines)
