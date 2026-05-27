"""
ProjectWorkspace — Shared blackboard for agent communication and state
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 2 of the Agentic System Implementation Plan.
The central blackboard where agents read/write state, post findings,
and communicate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.workspace.workspace")


@dataclass
class FileVersion:
    """A versioned file snapshot."""

    path: Path
    content: str
    version: int = 1
    author: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""


@dataclass
class ArchitectureDecision:
    """A recorded architecture decision."""

    id: str
    title: str
    decision: str
    rationale: str
    alternatives: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = ""


class ProjectWorkspace:
    """Central blackboard shared by all agents.

    Agents read from the workspace, compute, and write back.
    The workspace tracks all mutations with attribution for audit.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

        # File state
        self.files: dict[str, FileVersion] = {}

        # Decision log
        self.architectural_decisions: list[ArchitectureDecision] = []

        # Execution state
        self.completed_tasks: dict[str, Any] = {}

        # Test state
        self.test_results: dict[str, Any] = {}
        self.lint_results: dict[str, Any] = {}

        # Learning state
        self.knowledge: dict[str, Any] = {}

    def read_file(self, path: str) -> str | None:
        """Read the latest version of a file."""
        fv = self.files.get(path)
        return fv.content if fv else None

    def write_file(
        self, path: str, content: str, author: str = "unknown", message: str = ""
    ) -> FileVersion:
        """Write a new version of a file."""
        existing = self.files.get(path)
        v = (existing.version + 1) if existing else 1
        fv = FileVersion(
            path=Path(path), content=content, version=v, author=author, message=message
        )
        self.files[path] = fv
        return fv

    def record_decision(
        self, title: str, decision: str, rationale: str, author: str = ""
    ) -> ArchitectureDecision:
        """Record an architecture decision."""
        ad = ArchitectureDecision(
            id=f"ADR-{len(self.architectural_decisions) + 1:03d}",
            title=title,
            decision=decision,
            rationale=rationale,
            author=author,
        )
        self.architectural_decisions.append(ad)
        return ad

    def get_summary(self) -> str:
        """Get a summary of workspace state."""
        lines = [
            f"Workspace: {self.root}",
            f"Files modified: {len(self.files)}",
            f"Architecture decisions: {len(self.architectural_decisions)}",
            f"Tasks completed: {len(self.completed_tasks)}",
        ]
        return "\n".join(lines)
