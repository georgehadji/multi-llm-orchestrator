"""
Planner — Project-level planning and dependency management
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles task ordering, level grouping, and execution plan generation.
Decomposes topological concerns from the main Orchestrator mediator.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Task

logger = logging.getLogger("orchestrator.engine_core.project_planner")


class ProjectPlanner:
    """Handles topological sorting and execution planning for projects."""

    def __init__(self, dep_resolver: Any = None):
        self._dep_resolver = dep_resolver

    def get_execution_order(self, tasks: dict[str, Task]) -> list[str]:
        """Returns a deterministic topological sort of tasks."""
        if self._dep_resolver and hasattr(self._dep_resolver, "topological_sort"):
            sorted_tasks = dict(sorted(tasks.items()))
            self._dep_resolver.build_dependency_graph(sorted_tasks)
            try:
                return self._dep_resolver.topological_sort(sorted_tasks)
            except ValueError:
                return self._dep_resolver.execution_order
        
        # Fallback to local implementation if resolver not available
        return self._local_topological_sort(tasks)

    def get_execution_levels(self, tasks: dict[str, Task]) -> list[list[str]]:
        """Groups tasks into parallel execution levels."""
        in_degree = dict.fromkeys(tasks, 0)
        graph: dict[str, list[str]] = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        levels: list[list[str]] = []
        ready = sorted(tid for tid, deg in in_degree.items() if deg == 0)

        while ready:
            levels.append(ready)
            next_ready: list[str] = []
            for node in ready:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = sorted(next_ready)

        return levels

    def _local_topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """Simple Kahn's algorithm implementation."""
        levels = self.get_execution_levels(tasks)
        order = []
        for level in levels:
            order.extend(level)
        return order
