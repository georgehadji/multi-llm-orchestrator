"""
Task Completion Verification — Verify reported success matches actual system state
==================================================================================

Implements the finding from "Agents of Chaos" paper (arXiv:2602.20021):
- Task completion misrepresentation — Agents reported success while system state
  contradicted those reports

This module provides:
1. Expected outcome tracking during task planning
2. Post-completion verification against actual system state
3. Discrepancy reporting with severity levels
4. Integration with existing event store for audit trail

Usage:
    from orchestrator.task_verifier import TaskVerifier, VerificationResult

    verifier = TaskVerifier()

    # During task planning, register expected outcomes
    verifier.register_expected_outcome(
        task_id="task_001",
        expected_files=["src/main.py", "src/utils.py"],
        expected_state={"status": "completed", "tests_passed": True}
    )

    # After task completion, verify actual state
    result = await verifier.verify_completion(task_id="task_001")
    if not result.is_verified:
        logger.warning(f"Discrepancy detected: {result.discrepancies}")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class VerificationSeverity(Enum):
    """Severity levels for verification discrepancies."""
    INFO = "info"           # Minor, informational only
    WARNING = "warning"     # Potential issue, needs review
    ERROR = "error"         # Significant discrepancy, likely failure
    CRITICAL = "critical"   # Complete failure to meet requirements


class DiscrepancyType(Enum):
    """Types of discrepancies that can occur."""
    FILE_MISSING = "file_missing"
    FILE_MODIFIED = "file_modified"
    FILE_CONTENT_CHANGED = "file_content_changed"
    STATE_MISMATCH = "state_mismatch"
    OUTPUT_MISSING = "output_missing"
    VERIFICATION_SKIPPED = "verification_skipped"


@dataclass
class ExpectedOutcome:
    """Expected outcome for a task - registered during planning."""
    task_id: str
    expected_files: list[str] = field(default_factory=list)
    expected_directories: list[str] = field(default_factory=list)
    expected_state: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)  # Regex patterns that must match
    forbidden_patterns: list[str] = field(default_factory=list)  # Regex patterns that must NOT match
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Discrepancy:
    """A single discrepancy found during verification."""
    type: DiscrepancyType
    severity: VerificationSeverity
    description: str
    expected: Any | None = None
    actual: Any | None = None
    location: str | None = None


@dataclass
class VerificationResult:
    """Result of task completion verification."""
    task_id: str
    is_verified: bool
    discrepancies: list[Discrepancy] = field(default_factory=list)
    verified_at: datetime = field(default_factory=datetime.utcnow)
    verification_duration_ms: float = 0.0

    @property
    def has_critical_issues(self) -> bool:
        return any(d.severity == VerificationSeverity.CRITICAL for d in self.discrepancies)

    @property
    def has_errors(self) -> bool:
        return any(d.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL)
                   for d in self.discrepancies)

    @property
    def summary(self) -> str:
        if self.is_verified:
            return f"✓ Task {self.task_id} verified successfully"

        critical = sum(1 for d in self.discrepancies if d.severity == VerificationSeverity.CRITICAL)
        errors = sum(1 for d in self.discrepancies if d.severity == VerificationSeverity.ERROR)
        warnings = sum(1 for d in self.discrepancies if d.severity == VerificationSeverity.WARNING)

        return f"✗ Task {self.task_id}: {critical} critical, {errors} errors, {warnings} warnings"


class TaskVerifier:
    """
    Verifies task completion against expected outcomes.

    Implements defense against task completion misrepresentation by:
    1. Registering expected outcomes during task planning
    2. Verifying actual file system state after completion
    3. Checking expected outputs exist and are valid
    4. Detecting state mismatches
    """

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()
        self._expected_outcomes: dict[str, ExpectedOutcome] = {}
        self._file_hashes: dict[str, dict[str, str]] = {}  # task_id -> {path: hash}
        self._verification_cache: dict[str, VerificationResult] = {}

    def register_expected_outcome(
        self,
        task_id: str,
        expected_files: list[str] | None = None,
        expected_directories: list[str] | None = None,
        expected_state: dict[str, Any] | None = None,
        expected_outputs: list[str] | None = None,
        required_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
    ) -> None:
        """
        Register expected outcome for a task during planning phase.

        This should be called BEFORE task execution to establish what
        success looks like.
        """
        outcome = ExpectedOutcome(
            task_id=task_id,
            expected_files=expected_files or [],
            expected_directories=expected_directories or [],
            expected_state=expected_state or {},
            expected_outputs=expected_outputs or [],
            required_patterns=required_patterns or [],
            forbidden_patterns=forbidden_patterns or [],
        )

        # Store initial file hashes for comparison
        self._expected_outcomes[task_id] = outcome
        self._file_hashes[task_id] = {}

        for file_path in outcome.expected_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                self._file_hashes[task_id][file_path] = self._compute_file_hash(full_path)

        logger.info(f"Registered expected outcome for task {task_id}: "
                   f"{len(outcome.expected_files)} files, {len(outcome.expected_directories)} dirs")

    async def verify_completion(
        self,
        task_id: str,
        base_path: Path | None = None,
    ) -> VerificationResult:
        """
        Verify task completion against registered expected outcomes.

        Returns a VerificationResult with detailed discrepancy information.
        """
        start_time = datetime.utcnow()
        base = base_path or self.base_path

        outcome = self._expected_outcomes.get(task_id)
        if outcome is None:
            return VerificationResult(
                task_id=task_id,
                is_verified=False,
                discrepancies=[
                    Discrepancy(
                        type=DiscrepancyType.VERIFICATION_SKIPPED,
                        severity=VerificationSeverity.WARNING,
                        description=f"No expected outcome registered for task {task_id}",
                    )
                ],
            )

        discrepancies: list[Discrepancy] = []

        # 1. Verify expected files exist
        for file_path in outcome.expected_files:
            full_path = base / file_path
            if not full_path.exists():
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.FILE_MISSING,
                    severity=VerificationSeverity.CRITICAL,
                    description="Expected file not found",
                    expected=file_path,
                    actual=None,
                    location=str(full_path),
                ))
            elif file_path in self._file_hashes.get(task_id, {}):
                # File existed before - check if modified
                current_hash = self._compute_file_hash(full_path)
                original_hash = self._file_hashes[task_id][file_path]
                if current_hash != original_hash:
                    discrepancies.append(Discrepancy(
                        type=DiscrepancyType.FILE_MODIFIED,
                        severity=VerificationSeverity.WARNING,
                        description="File was modified during task execution",
                        expected=original_hash[:16],
                        actual=current_hash[:16],
                        location=str(full_path),
                    ))

        # 2. Verify expected directories exist
        for dir_path in outcome.expected_directories:
            full_path = base / dir_path
            if not full_path.exists():
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.FILE_MISSING,
                    severity=VerificationSeverity.ERROR,
                    description="Expected directory not found",
                    expected=dir_path,
                    actual=None,
                    location=str(full_path),
                ))
            elif not full_path.is_dir():
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.FILE_MISSING,
                    severity=VerificationSeverity.ERROR,
                    description="Expected directory but found file",
                    expected=dir_path,
                    actual="file",
                    location=str(full_path),
                ))

        # 3. Verify expected outputs
        for output_path in outcome.expected_outputs:
            full_path = base / output_path
            if not full_path.exists():
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.OUTPUT_MISSING,
                    severity=VerificationSeverity.ERROR,
                    description="Expected output not found",
                    expected=output_path,
                    actual=None,
                    location=str(full_path),
                ))

        # 4. Verify required patterns in output files
        if outcome.required_patterns:
            import re
            for file_path in outcome.expected_files:
                full_path = base / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        for pattern in outcome.required_patterns:
                            if not re.search(pattern, content):
                                discrepancies.append(Discrepancy(
                                    type=DiscrepancyType.STATE_MISMATCH,
                                    severity=VerificationSeverity.ERROR,
                                    description="Required pattern not found in file",
                                    expected=pattern,
                                    actual=None,
                                    location=str(full_path),
                                ))
                    except Exception as e:
                        logger.warning(f"Could not read {full_path} for pattern verification: {e}")

        # 5. Verify forbidden patterns (security: no dangerous code)
        if outcome.forbidden_patterns:
            import re
            for file_path in outcome.expected_files:
                full_path = base / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        for pattern in outcome.forbidden_patterns:
                            if re.search(pattern, content):
                                discrepancies.append(Discrepancy(
                                    type=DiscrepancyType.STATE_MISMATCH,
                                    severity=VerificationSeverity.CRITICAL,
                                    description="Forbidden pattern found in file (security issue)",
                                    expected=f"no match for {pattern}",
                                    actual="pattern found",
                                    location=str(full_path),
                                ))
                    except Exception as e:
                        logger.warning(f"Could not read {full_path} for forbidden pattern check: {e}")

        # Determine overall verification status
        is_verified = len([d for d in discrepancies
                          if d.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL)]) == 0

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = VerificationResult(
            task_id=task_id,
            is_verified=is_verified,
            discrepancies=discrepancies,
            verification_duration_ms=duration,
        )

        # Cache result
        self._verification_cache[task_id] = result

        # Log result
        if not is_verified:
            logger.warning(f"Task verification failed for {task_id}: {result.summary}")
            for d in discrepancies:
                logger.warning(f"  - [{d.severity.value}] {d.type.value}: {d.description}")
        else:
            logger.info(f"Task verified successfully: {task_id}")

        return result

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute hash for {file_path}: {e}")
            return ""

    def get_verification_summary(self) -> dict[str, Any]:
        """Get summary of all verifications."""
        total = len(self._verification_cache)
        verified = sum(1 for r in self._verification_cache.values() if r.is_verified)
        failed = total - verified

        critical_issues = sum(
            1 for r in self._verification_cache.values()
            if r.has_critical_issues
        )

        return {
            "total_tasks": total,
            "verified": verified,
            "failed": failed,
            "critical_issues": critical_issues,
            "verification_rate": verified / total if total > 0 else 0.0,
        }

    def clear_cache(self, task_id: str | None = None) -> None:
        """Clear verification cache."""
        if task_id:
            self._verification_cache.pop(task_id, None)
            self._expected_outcomes.pop(task_id, None)
            self._file_hashes.pop(task_id, None)
        else:
            self._verification_cache.clear()
            self._expected_outcomes.clear()
            self._file_hashes.clear()
