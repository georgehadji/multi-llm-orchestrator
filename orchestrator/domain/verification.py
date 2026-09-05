"""
Domain verification types — pure data, no I/O, no behavior beyond validation.

WBS-1: Mandatory acting verification.
CheckOutcome, ExecutionReceipt, VerificationPolicy — infrastructure-free types
used by the application-level VerificationGate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet


class CheckOutcome(str, Enum):
    """Structured outcome for a single verification check.

    Distinguishes four states (WBS-1):
        NOT_RUN  — check was skipped (by policy, feature flag, or missing tool).
        PASSED   — check ran and passed.
        FAILED   — check ran and found problems.
        BLOCKED  — check could not execute (tool missing, timeout, exception).
    """

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"

    # Requirement levels for VerificationPolicy
    OPTIONAL = "optional"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    MANDATORY = "mandatory"

    @property
    def is_passed(self) -> bool:
        return self == CheckOutcome.PASSED

    @property
    def is_failure(self) -> bool:
        return self in (CheckOutcome.FAILED, CheckOutcome.BLOCKED)


@dataclass(frozen=True)
class ExecutionReceipt:
    """Structured result of running a single verification check.

    Attributes:
        check_name:  Unique name for this check (e.g. "syntax", "lint", "tests").
        outcome:     PASSED, FAILED, BLOCKED, or NOT_RUN.
        reason:      Human-readable explanation (failure details, skip reason, …).
        duration_ms: Wall-clock execution time in milliseconds (None if not run).
        command:     Shell command or check label executed (for audit).
        artifact_hash: SHA-256 (or similar) of the artifact that was checked.
    """

    check_name: str
    outcome: CheckOutcome
    reason: str | None = None
    duration_ms: float | None = None
    command: str | None = None
    artifact_hash: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.outcome == CheckOutcome.BLOCKED

    @property
    def is_failure(self) -> bool:
        return self.outcome in (CheckOutcome.FAILED, CheckOutcome.BLOCKED)


@dataclass(frozen=True)
class VerificationPolicy:
    """Defines which verification checks are required for which task and artifact types.

    Attributes:
        checks:        Mapping of check name → requirement level (OPTIONAL, RECOMMENDED,
                       REQUIRED, MANDATORY).
        task_types:    If set, policy applies only to these TaskType values.
                       If None, the policy applies to all task types.
        artifact_types:If set, policy applies only to these artifact type identifiers.
                       If None, applies to all artifact types.
    """

    checks: Mapping[str, CheckOutcome] = field(default_factory=dict)
    task_types: FrozenSet[Any] | None = None
    artifact_types: FrozenSet[str] | None = None

    @property
    def required_checks(self) -> set[str]:
        """Return names of checks with REQUIRED or MANDATORY level."""
        return {
            name
            for name, level in self.checks.items()
            if level in (CheckOutcome.REQUIRED, CheckOutcome.MANDATORY)
        }

    @property
    def mandatory_checks(self) -> set[str]:
        """Return names of checks with MANDATORY level (fail-closed)."""
        return {name for name, level in self.checks.items() if level == CheckOutcome.MANDATORY}

    def get_requirement(self, check_name: str) -> CheckOutcome | None:
        """Return the requirement level for *check_name*, or None if unknown."""
        return self.checks.get(check_name)

    def applies_to(self, task_type: Any) -> bool:
        """Return True if this policy applies to *task_type*."""
        if self.task_types is None:
            return True
        return task_type in self.task_types

    def applies_to_artifact(self, artifact_type: str) -> bool:
        """Return True if this policy applies to *artifact_type*."""
        if self.artifact_types is None:
            return True
        return artifact_type in self.artifact_types


@dataclass(frozen=True)
class DeterministicResult:
    """Serialized gate result carried through CritiqueReport.

    Versioned for forward-compatibility across serialization boundaries
    (CritiqueReport.to_dict / from_dict).

    Attributes:
        passed:         All checks passed.
        checks:         name → passed (bool).
        reasons:        name → failure reason.
        artifact_hash:  SHA-256 hex digest of the artifact checked.
        failure_summary:{check_name: reason} for each failed/blocked check.
        status_summary: Human-readable one-liner of outcomes.
        version:        Schema version for forward-compat.
    """

    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    artifact_hash: str | None = None
    failure_summary: dict[str, str] = field(default_factory=dict)
    status_summary: str = "(no checks)"
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary including version marker."""
        return {
            "version": self.version,
            "passed": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
            "artifact_hash": self.artifact_hash,
            "failure_summary": self.failure_summary,
            "status_summary": self.status_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeterministicResult":
        """Deserialize from dictionary with version validation."""
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported DeterministicResult version={version}; expected 1")
        # Accept both old (unversioned) and new (versioned) dicts
        return cls(
            passed=data.get("passed", True),
            checks=data.get("checks", {}),
            reasons=data.get("reasons", {}),
            artifact_hash=data.get("artifact_hash"),
            failure_summary=data.get("failure_summary", {}),
            status_summary=data.get("status_summary", "(no checks)"),
        )
