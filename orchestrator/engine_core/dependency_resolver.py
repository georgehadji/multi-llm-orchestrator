"""
Dependency Resolver — DAG Resolution & Topological Sort
========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles task dependency resolution, topological sorting, and dependency context building.

Part of Engine Decomposition (Phase 1) - Extracted from engine.py
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Task, TaskResult

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Resolves task dependencies and determines execution order.

    Responsibilities:
    1. Build dependency graph from task definitions
    2. Perform topological sort for execution ordering
    3. Track completed tasks for dependency satisfaction
    4. Build context from dependency outputs
    """

    def __init__(self, context_truncation_limit: int = 40000):
        """
        Initialize dependency resolver.

        Args:
            context_truncation_limit: Max chars per dependency in context
        """
        self.dependency_graph: dict[str, list[str]] = defaultdict(list)
        self.reverse_graph: dict[str, list[str]] = defaultdict(list)
        self.execution_order: list[str] = []
        self.completed_tasks: set[str] = set()
        self.context_truncation_limit = context_truncation_limit

    def build_dependency_graph(self, tasks: dict[str, Task]) -> None:
        """
        Build dependency graph from task definitions.

        Args:
            tasks: Dictionary of task_id → Task
        """
        self.dependency_graph.clear()
        self.reverse_graph.clear()

        for task_id, task in tasks.items():
            # Ensure task exists in graph even with no dependencies
            if task_id not in self.dependency_graph:
                self.dependency_graph[task_id] = []

            # Add edges from dependencies to this task
            for dep_id in task.dependencies:
                self.dependency_graph[dep_id].append(task_id)
                self.reverse_graph[task_id].append(dep_id)

        logger.debug(f"Built dependency graph with {len(tasks)} tasks")

    def topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """
        Perform topological sort using Kahn's algorithm.

        FIX #6: Uses collections.deque instead of list.sort()+pop(0)
        for O(1) popleft instead of O(n) pop(0).

        Args:
            tasks: Dictionary of task_id → Task

        Returns:
            List of task IDs in execution order

        Raises:
            ValueError: If circular dependency detected
        """
        # Calculate in-degree for each task
        in_degree: dict[str, int] = dict.fromkeys(tasks, 0)

        for task_id, task in tasks.items():
            for dep_id in task.dependencies:
                if dep_id in tasks:
                    in_degree[task_id] += 1

        # Initialize queue with tasks that have no dependencies
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            task_id = queue.popleft()
            result.append(task_id)

            # Reduce in-degree for dependent tasks
            for dependent_task in self.dependency_graph[task_id]:
                if dependent_task in in_degree:
                    in_degree[dependent_task] -= 1
                    if in_degree[dependent_task] == 0:
                        queue.append(dependent_task)

        # Check for circular dependencies
        if len(result) != len(tasks):
            missing = set(tasks.keys()) - set(result)
            raise ValueError(f"Circular dependency detected involving tasks: {missing}")

        self.execution_order = result
        logger.debug(f"Topological sort complete: {len(result)} tasks")
        return result

    def get_ready_tasks(
        self,
        tasks: dict[str, Task],
        results: dict[str, TaskResult],
    ) -> list[str]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).

        Args:
            tasks: Dictionary of task_id → Task
            results: Dictionary of completed task results

        Returns:
            List of task IDs ready for execution
        """
        ready = []

        for task_id, task in tasks.items():
            # Skip if already completed
            if task_id in results:
                continue

            # Check if all dependencies are satisfied
            deps_satisfied = all(
                dep_id in results and results[dep_id].success
                for dep_id in task.dependencies
                if dep_id in tasks
            )

            if deps_satisfied:
                ready.append(task_id)

        return ready

    def mark_task_complete(self, task_id: str) -> None:
        """
        Mark a task as completed for dependency tracking.

        Args:
            task_id: ID of completed task
        """
        self.completed_tasks.add(task_id)
        logger.debug(f"Task {task_id} marked complete")

    def get_dependency_context(
        self,
        task: Task,
        results: dict[str, TaskResult],
        tasks: dict[str, Task],
    ) -> str:
        """
        Build context string from dependency outputs.

        Args:
            task: Task to build context for
            results: Dictionary of completed task results
            tasks: All tasks in project

        Returns:
            Formatted context string from dependencies
        """
        if not task.dependencies:
            return ""

        context_parts = []

        for dep_id in task.dependencies:
            if dep_id not in results or not results[dep_id].success:
                logger.warning(f"Dependency {dep_id} not available for {task.id}")
                continue

            dep_result = results[dep_id]
            dep_task = tasks.get(dep_id)

            # Build context section for this dependency
            context_section = self._format_dependency_context(dep_id, dep_task, dep_result)

            if context_section:
                context_parts.append(context_section)

        if not context_parts:
            return ""

        return "## Dependency Context\n\n" + "\n\n".join(context_parts)

    def _format_dependency_context(
        self,
        dep_id: str,
        dep_task: Task | None,
        dep_result: TaskResult,
    ) -> str:
        """
        Format a single dependency's context.

        Args:
            dep_id: Dependency task ID
            dep_task: Dependency task definition
            dep_result: Dependency execution result

        Returns:
            Formatted context section
        """
        output = dep_result.output or ""

        # Truncate if exceeds limit
        if len(output) > self.context_truncation_limit:
            truncated_len = self.context_truncation_limit
            output = output[:truncated_len] + "\n\n[... truncated ...]"
            logger.warning(
                f"Truncated dependency {dep_id} context from "
                f"{len(dep_result.output)} to {truncated_len} chars"
            )

        # Format based on task type
        task_type = dep_task.type.value if dep_task else "unknown"

        return f"### {dep_id} ({task_type})\n" f"```\n{output}\n```\n"

    def reset(self) -> None:
        """Reset resolver state for new project."""
        self.dependency_graph.clear()
        self.reverse_graph.clear()
        self.execution_order.clear()
        self.completed_tasks.clear()
        logger.debug("Dependency resolver reset")

    def get_dependents(self, task_id: str) -> list[str]:
        """
        Get all tasks that depend on the given task.

        Args:
            task_id: Task ID to find dependents for

        Returns:
            List of dependent task IDs
        """
        return list(self.dependency_graph.get(task_id, []))

    def get_dependencies(self, task_id: str) -> list[str]:
        """
        Get all tasks that the given task depends on.

        Args:
            task_id: Task ID to find dependencies for

        Returns:
            List of dependency task IDs
        """
        return list(self.reverse_graph.get(task_id, []))

    def is_dependency_satisfied(
        self,
        task_id: str,
        results: dict[str, TaskResult],
    ) -> bool:
        """
        Check if all dependencies for a task are satisfied.

        Args:
            task_id: Task to check
            results: Dictionary of completed task results

        Returns:
            True if all dependencies satisfied
        """
        if task_id not in self.reverse_graph:
            return True

        return all(
            dep_id in results and results[dep_id].success for dep_id in self.reverse_graph[task_id]
        )
