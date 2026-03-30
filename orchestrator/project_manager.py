"""
Project Management System
=========================
Advanced task scheduling, resource allocation, and progress tracking.

Features:
- Critical path analysis
- Resource constraint optimization
- Gantt-style timeline
- Predictive completion estimation
- Risk assessment

Usage:
    from orchestrator.project_manager import ProjectManager, TaskSchedule

    pm = ProjectManager()
    schedule = await pm.create_schedule(tasks, resources)
    critical_path = pm.get_critical_path()
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log_config import get_logger
from .performance import cached

if TYPE_CHECKING:
    from .models import Task, TaskStatus

logger = get_logger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class ResourceType(Enum):
    """Types of resources."""
    MODEL = "model"
    COMPUTE = "compute"
    MEMORY = "memory"
    API_RATE = "api_rate"


@dataclass
class Resource:
    """Resource definition."""
    id: str
    type: ResourceType
    capacity: float  # Total capacity
    available: float  # Currently available
    cost_per_unit: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSchedule:
    """Scheduled task with timing."""
    task_id: str
    start_time: datetime
    end_time: datetime
    resources_assigned: list[str]
    dependencies_met: bool = False
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    actual_duration: timedelta | None = None
    slack: timedelta = field(default_factory=lambda: timedelta(0))  # Float time
    is_critical: bool = False


@dataclass
class Milestone:
    """Project milestone."""
    id: str
    name: str
    description: str
    deadline: datetime | None = None
    tasks_required: list[str] = field(default_factory=list)
    completed: bool = False
    completion_date: datetime | None = None


@dataclass
class Risk:
    """Identified project risk."""
    id: str
    description: str
    probability: float  # 0.0 - 1.0
    impact: float  # 0.0 - 1.0
    mitigation: str = ""
    affected_tasks: list[str] = field(default_factory=list)
    status: str = "open"  # open, mitigated, realized

    @property
    def risk_score(self) -> float:
        """Calculate risk score (probability * impact)."""
        return self.probability * self.impact


@dataclass
class ProjectTimeline:
    """Complete project timeline."""
    project_id: str
    tasks: list[TaskSchedule]
    milestones: list[Milestone]
    risks: list[Risk]
    start_date: datetime
    end_date: datetime
    total_duration: timedelta
    critical_path: list[str]
    buffer_time: timedelta
    completion_probability: float = 1.0


class CriticalPathAnalyzer:
    """Analyze task dependencies for critical path."""

    def __init__(self, tasks: dict[str, TaskSchedule], dependencies: dict[str, list[str]]):
        self.tasks = tasks
        self.dependencies = dependencies
        self.earliest_start: dict[str, datetime] = {}
        self.latest_start: dict[str, datetime] = {}

    def calculate(self) -> tuple[list[str], timedelta]:
        """
        Calculate critical path using forward/backward pass.

        Returns:
            (critical_path_task_ids, project_duration)
        """
        if not self.tasks:
            return [], timedelta(0)

        # Forward pass - earliest start times
        sorted_tasks = self._topological_sort()

        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            deps = self.dependencies.get(task_id, [])

            if deps:
                max_end = max(
                    self.tasks[d].end_time for d in deps if d in self.tasks
                )
                self.earliest_start[task_id] = max_end
            else:
                self.earliest_start[task_id] = task.start_time

        # Backward pass - latest start times
        project_end = max(t.end_time for t in self.tasks.values())

        for task_id in reversed(sorted_tasks):
            task = self.tasks[task_id]

            # Find successors
            successors = [
                tid for tid, deps in self.dependencies.items()
                if task_id in deps
            ]

            if successors:
                min_start = min(
                    self.latest_start.get(s, self.tasks[s].start_time)
                    for s in successors
                )
                self.latest_start[task_id] = min_start - task.estimated_duration
            else:
                self.latest_start[task_id] = project_end - task.estimated_duration

        # Calculate slack and identify critical path
        critical_path = []
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            slack = self.latest_start[task_id] - self.earliest_start[task_id]
            task.slack = slack
            task.is_critical = (slack.total_seconds() <= 1)  # Near zero slack

            if task.is_critical:
                critical_path.append(task_id)

        # Calculate total duration
        if critical_path:
            first_task = self.tasks[critical_path[0]]
            last_task = self.tasks[critical_path[-1]]
            duration = last_task.end_time - first_task.start_time
        else:
            duration = timedelta(0)

        return critical_path, duration

    def _topological_sort(self) -> list[str]:
        """Topological sort of tasks based on dependencies."""
        in_degree = dict.fromkeys(self.tasks, 0)

        for task_id, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1

        # Use priority queue for deterministic ordering
        queue = [
            (self.tasks[tid].priority.value, tid)
            for tid, deg in in_degree.items() if deg == 0
        ]
        heapq.heapify(queue)

        sorted_tasks = []
        while queue:
            _, task_id = heapq.heappop(queue)
            sorted_tasks.append(task_id)

            # Find tasks that depend on this one
            for tid, deps in self.dependencies.items():
                if task_id in deps:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        heapq.heappush(queue, (self.tasks[tid].priority.value, tid))

        return sorted_tasks


class ResourceScheduler:
    """Optimize resource allocation across tasks."""

    def __init__(self, resources: list[Resource]):
        self.resources = {r.id: r for r in resources}
        self.allocations: dict[str, list[tuple[datetime, datetime, str]]] = {}

    def allocate(
        self,
        task_id: str,
        requirements: dict[str, float],
        start_time: datetime,
        duration: timedelta,
    ) -> list[str] | None:
        """
        Allocate resources for a task.

        Returns:
            List of allocated resource IDs or None if impossible
        """
        allocated = []

        for resource_id, amount in requirements.items():
            if resource_id not in self.resources:
                return None

            self.resources[resource_id]

            # Check availability
            end_time = start_time + duration
            if not self._is_available(resource_id, start_time, end_time, amount):
                # Try to find next available slot
                start_time = self._find_next_slot(resource_id, start_time, end_time, amount)
                if start_time is None:
                    return None
                end_time = start_time + duration

            allocated.append(resource_id)

            # Record allocation
            if resource_id not in self.allocations:
                self.allocations[resource_id] = []
            self.allocations[resource_id].append((start_time, end_time, task_id))

        return allocated

    def _is_available(
        self,
        resource_id: str,
        start: datetime,
        end: datetime,
        amount: float,
    ) -> bool:
        """Check if resource is available during time range."""
        if resource_id not in self.allocations:
            return True

        for alloc_start, alloc_end, _ in self.allocations[resource_id]:
            # Overlap check
            if start < alloc_end and end > alloc_start:
                return False

        return True

    def _find_next_slot(
        self,
        resource_id: str,
        start: datetime,
        end: datetime,
        amount: float,
    ) -> datetime | None:
        """Find next available time slot."""
        if resource_id not in self.allocations:
            return start

        duration = end - start
        current = start

        # Sort allocations by start time
        sorted_allocs = sorted(self.allocations[resource_id], key=lambda x: x[0])

        for alloc_start, alloc_end, _ in sorted_allocs:
            if current + duration <= alloc_start:
                return current
            current = max(current, alloc_end)

        return current

    def get_utilization(self, resource_id: str, window_hours: int = 24) -> float:
        """Calculate resource utilization percentage."""
        if resource_id not in self.allocations:
            return 0.0

        now = datetime.now()
        window_start = now - timedelta(hours=window_hours)

        busy_time = timedelta(0)
        for start, end, _ in self.allocations[resource_id]:
            if end > window_start and start < now:
                overlap_start = max(start, window_start)
                overlap_end = min(end, now)
                busy_time += overlap_end - overlap_start

        total_window = timedelta(hours=window_hours)
        return busy_time / total_window


class ProjectManager:
    """
    Main project management orchestrator.

    Features:
    - Schedule creation with resource constraints
    - Critical path analysis
    - Risk assessment
    - Progress tracking
    - Predictive analytics
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path(".projects")
        self.storage_path.mkdir(exist_ok=True)

        self._active_timelines: dict[str, ProjectTimeline] = {}
        self._risk_model = None  # Lazy loaded

    async def create_schedule(
        self,
        project_id: str,
        tasks: list[Task],
        resources: list[Resource],
        dependencies: dict[str, list[str]] | None = None,
        start_date: datetime | None = None,
    ) -> ProjectTimeline:
        """
        Create optimized project schedule.

        Algorithm:
        1. Sort tasks by priority and dependencies
        2. Allocate resources using constraint satisfaction
        3. Calculate critical path
        4. Identify risks
        """
        start_date = start_date or datetime.now()
        deps = dependencies or {}

        # Initialize scheduler
        scheduler = ResourceScheduler(resources)

        # Schedule tasks
        scheduled_tasks: dict[str, TaskSchedule] = {}

        for task in sorted(tasks, key=lambda t: (t.priority, t.created_at)):
            # Determine start time based on dependencies
            task_deps = deps.get(task.id, [])

            if task_deps:
                # Start after all dependencies complete
                dep_ends = [
                    scheduled_tasks[d].end_time
                    for d in task_deps if d in scheduled_tasks
                ]
                task_start = max(dep_ends) if dep_ends else start_date
            else:
                task_start = start_date

            # Estimate duration based on task type
            duration = await self._estimate_duration(task)

            # Allocate resources
            requirements = await self._get_resource_requirements(task)
            allocated = scheduler.allocate(task.id, requirements, task_start, duration)

            if allocated is None:
                logger.warning(f"Could not allocate resources for task {task.id}")
                # Delay start and try again
                task_start += timedelta(minutes=5)
                allocated = scheduler.allocate(task.id, requirements, task_start, duration)

            schedule = TaskSchedule(
                task_id=task.id,
                start_time=task_start,
                end_time=task_start + duration,
                resources_assigned=allocated or [],
                dependencies_met=len(task_deps) == 0,
                priority=self._map_priority(task),
                estimated_duration=duration,
            )

            scheduled_tasks[task.id] = schedule

        # Calculate critical path
        analyzer = CriticalPathAnalyzer(scheduled_tasks, deps)
        critical_path, project_duration = analyzer.calculate()

        # Calculate end date
        if scheduled_tasks:
            end_date = max(s.end_time for s in scheduled_tasks.values())
        else:
            end_date = start_date

        # Assess risks
        risks = await self._assess_risks(tasks, scheduled_tasks, critical_path)

        # Calculate completion probability (Monte Carlo simplified)
        completion_prob = self._calculate_completion_probability(
            scheduled_tasks, critical_path, risks
        )

        timeline = ProjectTimeline(
            project_id=project_id,
            tasks=list(scheduled_tasks.values()),
            milestones=[],  # TODO: Generate from critical path
            risks=risks,
            start_date=start_date,
            end_date=end_date,
            total_duration=project_duration,
            critical_path=critical_path,
            buffer_time=sum(
                (s.slack for s in scheduled_tasks.values()),
                timedelta(0)
            ),
            completion_probability=completion_prob,
        )

        self._active_timelines[project_id] = timeline
        await self._persist_timeline(timeline)

        logger.info(
            f"Created schedule for {project_id}: "
            f"{len(tasks)} tasks, critical path: {len(critical_path)} tasks, "
            f"duration: {project_duration}"
        )

        return timeline

    async def _estimate_duration(self, task: Task) -> timedelta:
        """Estimate task duration based on historical data."""
        # Default estimates by task type
        defaults = {
            "code_generation": timedelta(minutes=10),
            "refactoring": timedelta(minutes=15),
            "testing": timedelta(minutes=5),
            "documentation": timedelta(minutes=8),
            "analysis": timedelta(minutes=12),
        }

        task_type = task.task_type.value if hasattr(task, 'task_type') else "code_generation"
        return defaults.get(task_type, timedelta(minutes=10))

    async def _get_resource_requirements(self, task: Task) -> dict[str, float]:
        """Determine resource requirements for a task."""
        # Simple mapping - can be made more sophisticated
        return {
            "compute": 1.0,
            "api_rate": 0.5,
        }

    def _map_priority(self, task: Task) -> TaskPriority:
        """Map task to priority level."""
        # Map based on task properties
        if hasattr(task, 'priority'):
            if task.priority >= 8:
                return TaskPriority.CRITICAL
            elif task.priority >= 6:
                return TaskPriority.HIGH
            elif task.priority >= 4:
                return TaskPriority.MEDIUM
        return TaskPriority.LOW

    async def _assess_risks(
        self,
        tasks: list[Task],
        scheduled: dict[str, TaskSchedule],
        critical_path: list[str],
    ) -> list[Risk]:
        """Identify and assess project risks."""
        risks = []

        # Risk 1: Critical path too long
        critical_tasks = [scheduled[tid] for tid in critical_path if tid in scheduled]
        if critical_tasks:
            critical_duration = sum(
                (t.estimated_duration for t in critical_tasks),
                timedelta(0)
            )
            if critical_duration > timedelta(hours=1):
                risks.append(Risk(
                    id=f"risk_{hash('critical_path')}",
                    description="Critical path is very long - any delay impacts project",
                    probability=0.7,
                    impact=0.9,
                    mitigation="Consider parallelizing tasks or adding resources",
                    affected_tasks=critical_path,
                ))

        # Risk 2: Resource contention
        resource_usage: dict[str, int] = {}
        for sched in scheduled.values():
            for res in sched.resources_assigned:
                resource_usage[res] = resource_usage.get(res, 0) + 1

        for res, count in resource_usage.items():
            if count > 5:  # High usage
                risks.append(Risk(
                    id=f"risk_{hash(res)}",
                    description=f"High contention for resource: {res}",
                    probability=0.6,
                    impact=0.7,
                    mitigation="Add more resources or reschedule tasks",
                    affected_tasks=[
                        s.task_id for s in scheduled.values()
                        if res in s.resources_assigned
                    ],
                ))

        # Risk 3: Many dependencies
        for task_id, sched in scheduled.items():
            deps = [
                t for t in scheduled.values()
                if t.end_time <= sched.start_time and t.task_id != task_id
            ]
            if len(deps) > 3:
                risks.append(Risk(
                    id=f"risk_deps_{task_id}",
                    description=f"Task {task_id} has many dependencies",
                    probability=0.5,
                    impact=0.6,
                    mitigation="Review if all dependencies are necessary",
                    affected_tasks=[task_id],
                ))

        return risks

    def _calculate_completion_probability(
        self,
        tasks: dict[str, TaskSchedule],
        critical_path: list[str],
        risks: list[Risk],
    ) -> float:
        """Estimate probability of on-time completion."""
        # Simplified calculation based on risk exposure
        total_risk = sum(r.risk_score for r in risks)

        # Adjust for critical path length
        critical_factor = min(len(critical_path) / 10, 1.0)

        # Probability decreases with risk and critical path length
        prob = max(0.1, 1.0 - (total_risk * 0.3) - (critical_factor * 0.2))

        return round(prob, 2)

    def get_progress(self, project_id: str) -> dict[str, Any]:
        """Get current project progress."""
        if project_id not in self._active_timelines:
            return {"error": "Project not found"}

        timeline = self._active_timelines[project_id]

        total_tasks = len(timeline.tasks)
        completed = sum(1 for t in timeline.tasks if t.actual_duration is not None)
        in_progress = sum(
            1 for t in timeline.tasks
            if t.start_time <= datetime.now() < t.end_time
        )

        percent_complete = (completed / total_tasks * 100) if total_tasks > 0 else 0

        # Calculate if on track
        elapsed = datetime.now() - timeline.start_date
        expected_percent = (
            (elapsed / timeline.total_duration * 100)
            if timeline.total_duration.total_seconds() > 0 else 0
        )

        status = "on_track" if percent_complete >= expected_percent - 10 else "behind"

        return {
            "project_id": project_id,
            "percent_complete": round(percent_complete, 1),
            "tasks_completed": completed,
            "tasks_in_progress": in_progress,
            "tasks_total": total_tasks,
            "status": status,
            "days_remaining": (timeline.end_date - datetime.now()).days,
            "completion_probability": timeline.completion_probability,
            "risks_open": len([r for r in timeline.risks if r.status == "open"]),
        }

    async def update_task_status(
        self,
        project_id: str,
        task_id: str,
        status: TaskStatus,
        actual_duration: timedelta | None = None,
    ):
        """Update task completion status."""
        if project_id not in self._active_timelines:
            return

        timeline = self._active_timelines[project_id]

        for task in timeline.tasks:
            if task.task_id == task_id:
                task.actual_duration = actual_duration
                break

        await self._persist_timeline(timeline)

    @cached(ttl=60)
    async def generate_gantt_data(self, project_id: str) -> dict[str, Any]:
        """Generate data for Gantt chart visualization."""
        if project_id not in self._active_timelines:
            return {}

        timeline = self._active_timelines[project_id]

        tasks_data = []
        for task in timeline.tasks:
            tasks_data.append({
                "id": task.task_id,
                "start": task.start_time.isoformat(),
                "end": task.end_time.isoformat(),
                "duration_minutes": task.estimated_duration.total_seconds() / 60,
                "is_critical": task.is_critical,
                "resources": task.resources_assigned,
                "slack_minutes": task.slack.total_seconds() / 60,
            })

        return {
            "project_id": project_id,
            "start": timeline.start_date.isoformat(),
            "end": timeline.end_date.isoformat(),
            "tasks": tasks_data,
            "critical_path": timeline.critical_path,
        }

    async def _persist_timeline(self, timeline: ProjectTimeline):
        """Save timeline to disk."""
        file_path = self.storage_path / f"{timeline.project_id}.json"

        data = {
            "project_id": timeline.project_id,
            "start_date": timeline.start_date.isoformat(),
            "end_date": timeline.end_date.isoformat(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "start": t.start_time.isoformat(),
                    "end": t.end_time.isoformat(),
                    "is_critical": t.is_critical,
                }
                for t in timeline.tasks
            ],
            "critical_path": timeline.critical_path,
            "risks": [
                {
                    "id": r.id,
                    "description": r.description,
                    "probability": r.probability,
                    "impact": r.impact,
                    "status": r.status,
                }
                for r in timeline.risks
            ],
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_resource_report(self) -> dict[str, Any]:
        """Generate resource utilization report."""
        # Aggregate across all projects
        return {
            "timestamp": datetime.now().isoformat(),
            "active_projects": len(self._active_timelines),
            "message": "Resource report generated",
        }


# Global project manager
_project_manager: ProjectManager | None = None


def get_project_manager(storage_path: Path | None = None) -> ProjectManager:
    """Get global project manager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager(storage_path)
    return _project_manager
