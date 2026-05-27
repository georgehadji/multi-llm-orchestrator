"""SprintPlanner — Sprint creation, milestone tracking, progress."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("orchestrator.project.sprint_planner")


@dataclass
class TaskInfo:
    id: str
    description: str
    status: str = "pending"
    effort_hours: float = 1.0


@dataclass
class Milestone:
    id: str
    title: str
    goal: str
    task_ids: list[str] = field(default_factory=list)
    deadline: str = ""
    completed: bool = False


@dataclass
class Sprint:
    id: str
    goal: str
    milestones: list[Milestone] = field(default_factory=list)
    tasks: dict[str, TaskInfo] = field(default_factory=dict)


class SprintPlanner:
    def create_sprint(self, goal, task_ids):
        sprint = Sprint(id=f"sprint_{len(task_ids)}", goal=goal)
        for tid in task_ids:
            sprint.tasks[tid] = TaskInfo(id=tid, description=tid)
        ms = Milestone(id="ms_1", title="Core", goal=goal, task_ids=list(task_ids))
        sprint.milestones.append(ms)
        return sprint

    def get_progress(self, sprint):
        if not sprint.tasks:
            return 0.0
        done = sum(1 for t in sprint.tasks.values() if t.status == "done")
        return done / len(sprint.tasks)
