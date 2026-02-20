"""
DAG visualization for orchestrator task dependency graphs.
Pure-Python implementation — no external dependencies required.
"""
from __future__ import annotations
from typing import Optional
from .models import Task, TaskResult, TaskType


class DagRenderer:
    """
    Renders task dependency graph as Mermaid, ASCII, or reports.
    """

    _TYPE_COLORS: dict[TaskType, str] = {
        TaskType.CODE_GEN:     "#89B4FA",
        TaskType.CODE_REVIEW:  "#A6E3A1",
        TaskType.REASONING:    "#CBA6F7",
        TaskType.EVALUATE:     "#FAB387",
        TaskType.WRITING:      "#F38BA8",
        TaskType.DATA_EXTRACT: "#94E2D5",
        TaskType.SUMMARIZE:    "#F9E2AF",
    }

    def __init__(
        self,
        tasks: dict[str, Task],
        results: Optional[dict[str, TaskResult]] = None,
        truncation_limit: int = 40_000,
    ) -> None:
        self.tasks = tasks
        self.results = results or {}
        self.truncation_limit = truncation_limit

    def to_mermaid(self) -> str:
        """Mermaid flowchart format."""
        lines = ["flowchart TD"]
        for tid, task in self.tasks.items():
            label = f"{tid}\n[{task.type.value}]"
            color = self._TYPE_COLORS.get(task.type, "#CDD6F4")
            lines.append(f'    {tid}["{label}"]')
            lines.append(f"    style {tid} fill:{color},color:#1E1E2E")
        if self.tasks:
            lines.append("")
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                lines.append(f"    {dep} --> {tid}")
        return "\n".join(lines)

    def critical_path(self) -> list[str]:
        """Longest path through DAG using DP on topological order."""
        if not self.tasks:
            return []
        order = self._topological_order()
        dist: dict[str, int] = {tid: 0 for tid in self.tasks}
        pred: dict[str, Optional[str]] = {tid: None for tid in self.tasks}

        for tid in order:
            task = self.tasks[tid]
            for dep in task.dependencies:
                if dep in dist and dist[dep] + 1 > dist[tid]:
                    dist[tid] = dist[dep] + 1
                    pred[tid] = dep

        end = max(dist, key=lambda t: dist[t])
        path: list[str] = []
        cur: Optional[str] = end
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        return list(reversed(path))

    def to_ascii(self) -> str:
        """ASCII level-based diagram."""
        if not self.tasks:
            return "(empty graph)"
        levels = self._levels()
        lines: list[str] = []
        for level_idx, level_tasks in enumerate(levels):
            row = "  ".join(
                f"[{tid}:{self.tasks[tid].type.value[:4]}]"
                for tid in level_tasks
            )
            lines.append(f"L{level_idx}: {row}")
            if level_idx < len(levels) - 1:
                lines.append("       " + "  ".join("|" for _ in level_tasks))
        return "\n".join(lines)

    def dependency_report(self) -> str:
        """Context-size report per task."""
        lines = ["Dependency Context Report", "=" * 40]
        for tid, task in self.tasks.items():
            if not task.dependencies:
                continue
            lines.append(f"\n{tid} ({task.type.value})")
            for dep in task.dependencies:
                result = self.results.get(dep)
                if result is None:
                    lines.append(f"  ← {dep}: no result recorded")
                    continue
                size = len(result.output)
                truncated = size > self.truncation_limit
                trunc_note = (
                    f"  ⚠ TRUNCATED ({self.truncation_limit:,} / {size:,} chars)"
                    if truncated
                    else ""
                )
                lines.append(
                    f"  ← {dep}: {size:,} chars  "
                    f"score={result.score:.3f}  "
                    f"status={result.status.value}"
                    f"{trunc_note}"
                )
        return "\n".join(lines)

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm."""
        in_degree = {tid: len(self.tasks[tid].dependencies) for tid in self.tasks}
        queue = [t for t, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for tid, task in self.tasks.items():
                if node in task.dependencies:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
        return order

    def _levels(self) -> list[list[str]]:
        """Group tasks into parallelisable levels."""
        levels: dict[str, int] = {}
        for tid in self._topological_order():
            task = self.tasks[tid]
            dep_level = max(
                (levels[d] for d in task.dependencies if d in levels), default=-1
            )
            levels[tid] = dep_level + 1
        if not levels:
            return []
        max_level = max(levels.values())
        result: list[list[str]] = [[] for _ in range(max_level + 1)]
        for tid, lvl in levels.items():
            result[lvl].append(tid)
        return result
