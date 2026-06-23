"""
Supervisor models — pure data classes.

These dataclasses intentionally contain no I/O, no asyncio, and no behaviour,
honouring the ``models.py = pure data" architectural rule.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DirectiveSource = Literal["human", "agent"]


@dataclass(frozen=True)
class Directive:
    """A single request to the Supervisor, regardless of origin."""

    source: DirectiveSource
    text: str
    project_id: str = ""
    criteria: str = ""
    budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


LessonKind = Literal["task_failed", "validation_failed", "degraded", "error", "budget"]


@dataclass(frozen=True)
class Lesson:
    """A structured, append-only record of something that went wrong."""

    id: str
    session_id: str
    project_id: str
    task_type: str
    kind: LessonKind
    signal: str
    detail: str
    created_at: float


@dataclass
class SupervisorSession:
    """Persistent identity of a conversation / run session."""

    id: str
    created_at: float
    updated_at: float
    status: str
    summary: str
    directive_count: int = 0


@dataclass(frozen=True)
class SupervisorResult:
    """Outcome returned after a Supervisor handle() call."""

    session_id: str
    project_status: str
    lessons_recorded: int
    # Reserved: the engine's streaming API does not expose the output location
    # and run_project_streaming() takes no output_dir. This is populated only
    # once the supervisor controls the engine output path (a later phase).
    output_dir: str | None = None


@dataclass(frozen=True)
class JobArgs:
    """Engine-ready arguments produced from a Directive by intake."""

    project_description: str
    success_criteria: str
    budget: float | None
    project_id: str
