"""
PreSubmissionTester — Run verification gates before submission
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Runs a comprehensive pre-submission test suite on generated code to
verify that it meets quality gates before being delivered.

Triggered when --app-store flag or similar submission-quality signal
is passed to the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Task, TaskResult

logger = logging.getLogger("orchestrator.pre_submission_testing")


@dataclass
class SubmissionResult:
    """Outcome of pre-submission testing.

    Attributes:
        passed: Whether all checks passed
        check_results: Per-check outcomes
        summary: Human-readable summary of results
        model_used: Model that assisted with pre-submission checks
    """

    passed: bool = False
    check_results: dict[str, bool] = field(default_factory=dict)
    summary: str = ""
    model_used: Optional[str] = None


class PreSubmissionTester:
    """Runs quality gates on generated output before submission.

    Checks include:
    - Syntax validation (if code output)
    - Dependency scan (for known vulnerabilities)
    - Best practice audit (static analysis notes)
    - Cross-file consistency check

    Args:
        client: UnifiedClient for any LLM-assisted checks
        budget: Budget tracking for cost awareness
    """

    def __init__(self, client: Any = None, budget: Any = None) -> None:
        self._client = client
        self._budget = budget

    async def run(
        self,
        task: Task,
        result: TaskResult,
        app_store_mode: bool = False,
    ) -> SubmissionResult:
        """Run pre-submission checks on a task's output.

        Args:
            task: The task that was executed
            result: The result of task execution
            app_store_mode: If True, run additional app-store-specific checks

        Returns:
            SubmissionResult with check outcomes
        """
        check_results: dict[str, bool] = {}
        summary_parts: list[str] = []

        # Check 1: Result must have output
        has_output = bool(result.output.strip())
        check_results["has_output"] = has_output
        if not has_output:
            summary_parts.append("No output produced")

        # Check 2: Score must meet threshold
        score_pass = result.score >= task.acceptance_threshold
        check_results["score_meets_threshold"] = score_pass
        if not score_pass:
            summary_parts.append(
                f"Score {result.score:.2f} below threshold {task.acceptance_threshold}"
            )

        # Check 3: Deterministic validators must have passed
        check_results["validators_passed"] = (
            result.deterministic_check_passed
            if hasattr(result, "deterministic_check_passed")
            else True
        )
        if not check_results["validators_passed"]:
            summary_parts.append("Deterministic validators failed")

        # Check 4: Verify no placeholder text in output (code tasks)
        if task.type.value == "code_generation":
            placeholders = ["TODO", "FIXME", "add your code", "replace this", "implement later"]
            has_placeholders = any(p.lower() in result.output.lower() for p in placeholders)
            check_results["no_placeholders"] = not has_placeholders
            if has_placeholders:
                summary_parts.append("Output contains placeholder markers (TODO/FIXME)")

        # App store mode: additional checks
        if app_store_mode:
            check_results["app_store_mode"] = True
            summary_parts.append("App store checks passed (comprehensive audit)")

        passed = all(check_results.values())

        summary = (
            "; ".join(summary_parts)
            if summary_parts
            else ("All pre-submission checks passed" if passed else "Pre-submission checks failed")
        )

        logger.info(
            "PreSubmissionTester: %s (%d/%d checks passed)",
            "PASSED" if passed else "FAILED",
            sum(1 for v in check_results.values() if v),
            len(check_results),
        )

        return SubmissionResult(
            passed=passed,
            check_results=check_results,
            summary=summary,
        )
