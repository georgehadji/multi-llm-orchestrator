"""
Tests for two critical regressions in resilience fixes (PR #5):

Regression #1: Reviewer circuit breaker not reset on success
- _record_failure() called on critique exception
- _record_success() NOT called on critique success
- Counter never resets, so 3 transient errors across long run permanently disable reviewer
- Breaks circuit breaker design: "consecutive" failures

Regression #2: COMPLETED_DEGRADED assumes full task execution
- COMPLETED_DEGRADED checks all_passed/det_ok from state.results
- state.results is sparse (early termination leaves tasks unexecuted)
- Partial execution can be marked COMPLETED_DEGRADED (terminal, not resumable)
- Next run skips _resume_project, leaves unfinished tasks permanently unexecuted
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from orchestrator.models import (
    Model, TaskType, Task, TaskStatus, TaskResult, ProjectState,
    ProjectStatus, Budget
)
from orchestrator.engine import Orchestrator


class TestReviewerCircuitBreakerReset:
    """Regression #1: Reviewer circuit breaker must reset on successful critique"""

    def test_successful_critique_resets_reviewer_failure_counter(self):
        """
        FAILING TEST: Successful critique should call _record_success to reset counter.

        Currently fails because critique success path never calls _record_success(reviewer).
        After fix: Counter resets on success, allowing recovery from transient errors.
        """
        orch = Orchestrator()
        reviewer = Model.CLAUDE_SONNET

        # Setup: healthy state
        orch.api_health[reviewer] = True
        orch._consecutive_failures[reviewer] = 0

        # Simulate 2 transient critique errors
        orch._record_failure(reviewer, error=Exception("429 rate limit"))
        orch._record_failure(reviewer, error=Exception("timeout"))
        assert orch._consecutive_failures.get(reviewer, 0) == 2
        assert orch.api_health.get(reviewer, True) is True  # Not yet disabled

        # Now simulate successful critique - must call _record_success
        mock_response = MagicMock()
        mock_response.latency_ms = 150
        mock_response.cost_usd = 0.005
        mock_response.input_tokens = 200
        mock_response.output_tokens = 100

        orch._record_success(reviewer, mock_response)

        # Counter should be reset after success
        assert orch._consecutive_failures.get(reviewer, 0) == 0, \
            "Success should reset consecutive failures counter for reviewer"

        # Now 2 more transient errors should still leave reviewer healthy
        orch._record_failure(reviewer, error=Exception("429"))
        orch._record_failure(reviewer, error=Exception("timeout"))
        assert orch.api_health.get(reviewer, True) is True, \
            "After reset, 2 failures should not disable reviewer (threshold is 3)"


    def test_three_transient_critique_errors_disable_reviewer_permanently(self):
        """
        Verify that 3 consecutive transient errors DO trigger circuit breaker.

        This ensures we're testing the full behavior: success resets, 3 failures disable.
        """
        orch = Orchestrator()
        reviewer = Model.GPT_4O

        orch.api_health[reviewer] = True
        orch._consecutive_failures[reviewer] = 0

        # Three transient failures without success between them
        orch._record_failure(reviewer, error=Exception("429"))
        orch._record_failure(reviewer, error=Exception("timeout"))
        orch._record_failure(reviewer, error=Exception("temporary"))

        # Should be permanently disabled
        assert orch.api_health.get(reviewer, True) is False, \
            "3 consecutive transient failures should trigger circuit breaker"


class TestCompletedDegradedGuard:
    """Regression #2: COMPLETED_DEGRADED must guard against partial execution"""

    def test_completed_degraded_requires_all_tasks_executed(self):
        """
        FAILING TEST: COMPLETED_DEGRADED should only be returned when ALL tasks executed.

        Currently fails because _determine_final_status checks all_passed/det_ok
        from sparse state.results without verifying len(state.results) == len(state.tasks).

        After fix: Partial execution will return PARTIAL_SUCCESS (resumable),
        only full execution can return COMPLETED_DEGRADED (terminal).
        """
        orch = Orchestrator()

        # Create a project with 5 tasks
        state = ProjectState(
            project_description="Test project",
            success_criteria="All pass",
            budget=Budget(max_usd=100.0),
        )

        # Add 5 tasks to the project
        for i in range(5):
            state.tasks[f"task_{i}"] = Task(
                id=f"task_{i}",
                type=TaskType.CODE_GEN,
                prompt=f"Task {i}"
            )

        # Execute only first 3 tasks (simulate early termination)
        for i in range(3):
            state.results[f"task_{i}"] = TaskResult(
                task_id=f"task_{i}",
                output=f"Output {i}",
                score=0.9,
                model_used=Model.CLAUDE_SONNET,
                status=TaskStatus.COMPLETED,
                deterministic_check_passed=False  # ← Failed validation
            )

        # Tasks 4-5 never executed (no results)
        assert len(state.results) == 3
        assert len(state.tasks) == 5

        # Determine final status
        status = orch._determine_final_status(state)

        # Must return PARTIAL_SUCCESS (resumable), NOT COMPLETED_DEGRADED (terminal)
        assert status == ProjectStatus.PARTIAL_SUCCESS, \
            "Partial execution with failed validation must return PARTIAL_SUCCESS (resumable), " \
            "not COMPLETED_DEGRADED (terminal)"


    def test_completed_degraded_only_when_all_tasks_executed_and_failed_validation(self):
        """
        Verify correct behavior: COMPLETED_DEGRADED only when all tasks executed AND some failed validation.
        """
        orch = Orchestrator()

        state = ProjectState(
            project_description="Test project",
            success_criteria="All pass",
            budget=Budget(max_usd=100.0),
        )

        # 3 tasks, all executed
        for i in range(3):
            state.tasks[f"task_{i}"] = Task(
                id=f"task_{i}",
                type=TaskType.CODE_GEN,
                prompt=f"Task {i}"
            )
            state.results[f"task_{i}"] = TaskResult(
                task_id=f"task_{i}",
                output=f"Output {i}",
                score=0.9,
                model_used=Model.CLAUDE_SONNET,
                status=TaskStatus.COMPLETED,
                deterministic_check_passed=False  # ← Failed validation
            )

        # All tasks executed, some failed validation
        assert len(state.results) == len(state.tasks) == 3

        status = orch._determine_final_status(state)

        # Should return COMPLETED_DEGRADED (terminal)
        assert status == ProjectStatus.COMPLETED_DEGRADED, \
            "All tasks executed with failed validation should return COMPLETED_DEGRADED (terminal)"


    def test_partial_execution_always_resumable(self):
        """
        Verify that any partial execution returns PARTIAL_SUCCESS (resumable),
        regardless of validation results on executed tasks.
        """
        orch = Orchestrator()

        state = ProjectState(
            project_description="Test",
            success_criteria="Pass",
            budget=Budget(max_usd=100.0),
        )

        # 10 tasks defined
        for i in range(10):
            state.tasks[f"t_{i}"] = Task(
                id=f"t_{i}",
                type=TaskType.CODE_GEN,
                prompt=f"Task {i}"
            )

        # Only 5 executed
        for i in range(5):
            state.results[f"t_{i}"] = TaskResult(
                task_id=f"t_{i}",
                output=f"Output {i}",
                score=0.95,
                model_used=Model.CLAUDE_SONNET,
                status=TaskStatus.COMPLETED,
                deterministic_check_passed=True  # ← Passed validation
            )

        status = orch._determine_final_status(state)

        # Even though executed tasks passed validation, partial execution = resumable
        assert status == ProjectStatus.PARTIAL_SUCCESS, \
            "Partial execution must always return PARTIAL_SUCCESS (resumable), " \
            "never COMPLETED_DEGRADED (terminal)"
