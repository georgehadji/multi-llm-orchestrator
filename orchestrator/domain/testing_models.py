"""
Testing domain models — pure data, no I/O, no asyncio (Contract 1).
=====================================================================
Author: Implementation Plan (Autonomous Testing Engine)

Phase 0 scaffold. These types form the vocabulary for the entire
testing subsystem and must be usable from every layer via imports
of domain types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TestStatus(str, Enum):
    """Outcome of a single test node."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"


class IsolationLevel(str, Enum):
    """Recorded on every report so downstream gates know what protection applied."""

    NONE = "none"  # never selectable — reserved for legacy receipts
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class CheckScope(str, Enum):
    """Describes what a verification check requires to execute."""

    ARTIFACT = "artifact"  # existing checks — unchanged, default
    WORKSPACE = "workspace"  # needs a materialized tree


@dataclass(frozen=True)
class TestOutcome:
    """Result of a single test node."""

    node_id: str
    status: TestStatus
    duration_ms: float
    message: str = ""


@dataclass(frozen=True)
class SuiteReport:
    """Structured result of one suite execution. Never derived from regex."""

    passed: bool
    exit_code: int
    outcomes: tuple[TestOutcome, ...] = ()
    collection_errors: tuple[str, ...] = ()  # distinct from failures — not LLM-repairable
    line_coverage: float | None = None
    mutation_score: float | None = None
    flaky_node_ids: tuple[str, ...] = ()
    isolation: IsolationLevel = IsolationLevel.SUBPROCESS
    duration_ms: float = 0.0
    truncated_output: str = ""

    @property
    def executed(self) -> int:
        return len(self.outcomes)

    @property
    def is_vacuous_result(self) -> bool:
        """Zero executed tests is never success — closes the D-7 false-positive."""
        return self.executed == 0


@dataclass(frozen=True)
class Workspace:
    """A materialized, multi-file project tree ready for execution."""

    root: Path
    framework: str
    source_files: tuple[Path, ...] = ()
    test_files: tuple[Path, ...] = ()
    manifest: Path | None = None  # pyproject.toml / package.json / go.mod
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TestSelection:
    """Select test subset to run. Empty node_ids means full suite."""

    node_ids: tuple[str, ...] = ()
    reason: str = "full"


__all__ = [
    "CheckScope",
    "IsolationLevel",
    "SuiteReport",
    "TestOutcome",
    "TestSelection",
    "TestStatus",
    "Workspace",
]
