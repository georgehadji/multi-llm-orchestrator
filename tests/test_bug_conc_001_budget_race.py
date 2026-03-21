"""
Test Suite: Budget Race Condition Fix (BUG-CONC-001)
=====================================================
Tests for atomic budget reserve pattern to prevent concurrent overcommitment.

Test Framework: pytest + pytest-asyncio
Coverage: Regression, Edge Cases, Failure Injection, Integration
"""
import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from orchestrator.models import Budget, Task, TaskType, TaskResult, TaskStatus, Model
from orchestrator.engine import Orchestrator


# ═══════════════════════════════════════════════════════════════════════════════
# 6a. REGRESSION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetRaceConditionRegression:
    """
    Regression tests for BUG-CONC-001: Budget race condition in concurrent execution.
    
    Original Bug: Multiple concurrent tasks check budget simultaneously, all pass,
    then all execute causing budget overcommitment.
    """

    @pytest.mark.asyncio
    async def test_concurrent_budget_checks_dont_overcommit(self):
        """
        TEST-ID: BUG-CONC-001-REG-01
        
        DESCRIPTION: Verify that concurrent task execution doesn't exceed budget
        even when multiple tasks check budget simultaneously.
        
        PRECONDITIONS:
        - Budget with $0.10 remaining
        - 5 tasks each costing ~$0.02
        
        INPUT:
        - 5 concurrent tasks in same dependency level
        - Each task will charge $0.02 when executed
        
        EXPECTED BEHAVIOR (with patch):
        - Only 5 tasks should execute (5 × $0.02 = $0.10)
        - Budget should never go negative
        - Remaining budget should be >= 0 after all tasks
        
        FAILURE BEHAVIOR (without patch):
        - All 5 tasks pass budget check simultaneously
        - All 5 execute and charge budget
        - Budget goes negative (e.g., -$0.02 to -$0.10)
        """
        # Arrange: Create budget with exact amount for 5 tasks
        budget = Budget(max_usd=0.10)
        orch = Orchestrator(budget=budget, max_parallel_tasks=5)
        
        # Mock _execute_task to charge exactly $0.02
        async def mock_execute(task):
            # Simulate API call cost
            orch.budget.charge(0.02, "generation")
            return TaskResult(
                task_id=task.id,
                output="code",
                score=0.9,
                model_used=Model.GPT_4O_MINI,
                status=TaskStatus.COMPLETED,
            )
        
        orch._execute_task = mock_execute
        
        # Create 5 tasks at same dependency level
        tasks = {
            f"task_{i}": Task(
                id=f"task_{i}",
                type=TaskType.CODE_GEN,
                prompt=f"Generate code for task {i}",
                dependencies=[],
            )
            for i in range(5)
        }
        orch.results = {}
        
        # Act: Execute all tasks concurrently
        await orch._execute_all(
            tasks=tasks,
            execution_order=[f"task_{i}" for i in range(5)],
            project_desc="Test",
            success_criteria="Test",
        )
        
        # Assert: Budget should not be negative
        assert budget.remaining_usd >= 0, f"Budget overcommitted: {budget.remaining_usd}"
        # All tasks should have executed (5 × $0.02 = $0.10 exactly)
        assert budget.spent_usd <= 0.10, f"Budget exceeded: {budget.spent_usd}"

    @pytest.mark.asyncio
    async def test_reservation_released_on_task_failure(self):
        """
        TEST-ID: BUG-CONC-001-REG-02
        
        DESCRIPTION: Verify that budget reservation is released if task fails
        before charging actual cost.
        
        PRECONDITIONS:
        - Budget with $0.10
        - Task reserves $0.02 but fails before charging
        
        INPUT:
        - Task that raises exception during _execute_task()
        
        EXPECTED BEHAVIOR (with patch):
        - Reservation is released in finally block
        - Budget remains available for other tasks
        - Remaining budget = $0.10 (no leak)
        
        FAILURE BEHAVIOR (without patch):
        - Reservation is never released
        - Budget._reserved_usd stays at $0.02
        - Remaining budget = $0.08 (leaked)
        """
        # Arrange
        budget = Budget(max_usd=0.10)
        orch = Orchestrator(budget=budget)
        
        # Mock task that fails
        async def mock_execute_fail(task):
            raise Exception("Task failed")
        
        orch._execute_task = mock_execute_fail
        
        tasks = {
            "task_1": Task(
                id="task_1",
                type=TaskType.CODE_GEN,
                prompt="Fail me",
                dependencies=[],
            )
        }
        orch.results = {}
        
        # Act: Execute failing task
        try:
            await orch._execute_all(
                tasks=tasks,
                execution_order=["task_1"],
                project_desc="Test",
                success_criteria="Test",
            )
        except Exception:
            pass  # Expected to fail
        
        # Assert: Budget should not have leaked reservation
        # (This test will fail until try/finally is implemented in FIX-001a)
        assert budget._reserved_usd == 0, f"Reservation leaked: {budget._reserved_usd}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetReserveEdgeCases:
    """
    Edge case tests for Budget.reserve() pattern.
    """

    @pytest.mark.asyncio
    async def test_reserve_exact_remaining_amount(self):
        """
        TEST-ID: BUG-CONC-001-EDGE-01
        
        DESCRIPTION: Reserve exactly the remaining budget amount.
        
        PRECONDITIONS:
        - Budget with $0.05 remaining
        
        INPUT:
        - reserve(0.05)
        
        EXPECTED BEHAVIOR (with patch):
        - Returns True
        - Remaining budget = $0.00
        
        FAILURE BEHAVIOR (without patch):
        - Floating point precision issues
        - May return False incorrectly
        """
        budget = Budget(max_usd=0.10)
        budget.charge(0.05, "test")  # Spend $0.05
        
        # Act: Reserve exact remaining
        result = budget.reserve(0.05)
        
        # Assert
        assert result is True
        assert budget.remaining_usd == 0.0

    @pytest.mark.asyncio
    async def test_reserve_exceeds_remaining(self):
        """
        TEST-ID: BUG-CONC-001-EDGE-02
        
        DESCRIPTION: Reserve more than remaining budget.
        
        PRECONDITIONS:
        - Budget with $0.05 remaining
        
        INPUT:
        - reserve(0.10)
        
        EXPECTED BEHAVIOR (with patch):
        - Returns False
        - No reservation made
        
        FAILURE BEHAVIOR (without patch):
        - May return True incorrectly
        - Budget overcommitted
        """
        budget = Budget(max_usd=0.10)
        budget.charge(0.05, "test")
        
        # Act
        result = budget.reserve(0.10)
        
        # Assert
        assert result is False
        assert budget._reserved_usd == 0

    @pytest.mark.asyncio
    async def test_reserve_zero_amount(self):
        """
        TEST-ID: BUG-CONC-001-EDGE-03
        
        DESCRIPTION: Reserve zero amount (edge case).
        
        PRECONDITIONS:
        - Budget with any remaining amount
        
        INPUT:
        - reserve(0.0)
        
        EXPECTED BEHAVIOR (with patch):
        - Returns True (zero reservation is always valid)
        - No impact on budget
        
        FAILURE BEHAVIOR (without patch):
        - May raise division by zero or validation error
        """
        budget = Budget(max_usd=0.10)
        
        # Act
        result = budget.reserve(0.0)
        
        # Assert
        assert result is True
        assert budget._reserved_usd == 0.0

    @pytest.mark.asyncio
    async def test_reserve_negative_amount(self):
        """
        TEST-ID: BUG-CONC-001-EDGE-04
        
        DESCRIPTION: Reserve negative amount (invalid input).
        
        PRECONDITIONS:
        - Budget with any remaining amount
        
        INPUT:
        - reserve(-0.01)
        
        EXPECTED BEHAVIOR (with patch):
        - Raises ValueError
        - No reservation made
        
        FAILURE BEHAVIOR (without patch):
        - May accept negative reservation
        - Budget logic corrupted
        """
        budget = Budget(max_usd=0.10)
        
        # Act + Assert
        with pytest.raises(ValueError):
            budget.reserve(-0.01)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_reservations(self):
        """
        TEST-ID: BUG-CONC-001-EDGE-05
        
        DESCRIPTION: Multiple concurrent reservations at budget limit.
        
        PRECONDITIONS:
        - Budget with $0.10 remaining
        - 10 coroutines each trying to reserve $0.02
        
        INPUT:
        - 10 concurrent reserve(0.02) calls
        
        EXPECTED BEHAVIOR (with patch):
        - Exactly 5 succeed (5 × $0.02 = $0.10)
        - 5 fail (budget exhausted)
        - Total reserved = $0.10
        
        FAILURE BEHAVIOR (without patch):
        - All 10 may succeed (race condition)
        - Total reserved = $0.20 (overcommitted)
        """
        budget = Budget(max_usd=0.10)
        results = []
        
        async def try_reserve():
            result = budget.reserve(0.02)
            results.append(result)
            return result
        
        # Act: 10 concurrent reservations
        await asyncio.gather(*[try_reserve() for _ in range(10)])
        
        # Assert
        success_count = sum(1 for r in results if r)
        assert success_count == 5, f"Expected 5 successes, got {success_count}"
        assert budget._reserved_usd == 0.10


# ═══════════════════════════════════════════════════════════════════════════════
# 6c. FAILURE INJECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetFailureInjection:
    """
    Failure injection tests for BUG-CONC-001 fix.
    """

    @pytest.mark.asyncio
    async def test_exception_during_execute_releases_reservation(self):
        """
        TEST-ID: BUG-CONC-001-FAIL-01
        
        DESCRIPTION: Inject exception during task execution, verify reservation cleanup.
        
        PRECONDITIONS:
        - Budget with $0.10
        - Task reserves $0.02
        
        INPUT:
        - Exception raised in middle of _execute_task()
        
        EXPECTED BEHAVIOR (with patch):
        - Exception propagates to caller
        - Reservation released in finally block
        - Budget available for retry
        
        FAILURE BEHAVIOR (without patch):
        - Reservation leaked
        - Budget._reserved_usd = $0.02 permanently
        """
        budget = Budget(max_usd=0.10)
        orch = Orchestrator(budget=budget)
        
        exception_raised = False
        
        async def mock_execute_with_exception(task):
            nonlocal exception_raised
            exception_raised = True
            raise ValueError("Injected failure")
        
        orch._execute_task = mock_execute_with_exception
        
        tasks = {"task_1": Task(id="task_1", type=TaskType.CODE_GEN, prompt="Test", dependencies=[])}
        orch.results = {}
        
        # Act: Execute task that will fail
        try:
            await orch._execute_all(
                tasks=tasks,
                execution_order=["task_1"],
                project_desc="Test",
                success_criteria="Test",
            )
        except ValueError:
            pass  # Expected
        
        # Assert
        assert exception_raised, "Exception was not raised"
        assert budget._reserved_usd == 0, f"Reservation leaked: {budget._reserved_usd}"

    @pytest.mark.asyncio
    async def test_timeout_during_execute_releases_reservation(self):
        """
        TEST-ID: BUG-CONC-001-FAIL-02
        
        DESCRIPTION: Inject timeout during task execution.
        
        PRECONDITIONS:
        - Budget with $0.10
        - Task with 10 second timeout
        
        INPUT:
        - asyncio.wait_for() timeout after 0.1 seconds
        
        EXPECTED BEHAVIOR (with patch):
        - TimeoutError raised
        - Reservation released
        - No memory leak
        
        FAILURE BEHAVIOR (without patch):
        - Task cancelled but reservation not released
        - Memory leak in long-running process
        """
        budget = Budget(max_usd=0.10)
        orch = Orchestrator(budget=budget)
        
        async def mock_execute_slow(task):
            await asyncio.sleep(10)  # Will timeout
            return TaskResult(task_id=task.id, output="", score=0.0, model_used=Model.GPT_4O_MINI)
        
        orch._execute_task = mock_execute_slow
        
        tasks = {"task_1": Task(id="task_1", type=TaskType.CODE_GEN, prompt="Test", dependencies=[])}
        orch.results = {}
        
        # Act: Execute with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                orch._execute_all(
                    tasks=tasks,
                    execution_order=["task_1"],
                    project_desc="Test",
                    success_criteria="Test",
                ),
                timeout=0.1,
            )
        
        # Assert
        assert budget._reserved_usd == 0, f"Reservation leaked on timeout: {budget._reserved_usd}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6d. INTEGRATION SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetReserveIntegration:
    """
    Integration smoke tests for Budget.reserve() with callers.
    """

    @pytest.mark.asyncio
    async def test_reserve_with_run_project(self):
        """
        TEST-ID: BUG-CONC-001-INT-01
        
        DESCRIPTION: Verify reserve pattern integrates with run_project().
        
        PRECONDITIONS:
        - Orchestrator with budget
        - Valid project description
        
        INPUT:
        - run_project() with small project
        
        EXPECTED BEHAVIOR (with patch):
        - Project executes normally
        - Budget reservations properly managed
        - No reservation leaks after completion
        
        FAILURE BEHAVIOR (without patch):
        - Budget overcommitment during concurrent task execution
        """
        budget = Budget(max_usd=1.00)
        orch = Orchestrator(budget=budget)
        
        # Mock decomposition to return simple task
        async def mock_decompose(project, criteria, **kwargs):
            return {
                "task_1": Task(
                    id="task_1",
                    type=TaskType.CODE_GEN,
                    prompt="Write hello world",
                    dependencies=[],
                )
            }
        
        orch._decompose = mock_decompose
        orch._execute_task = AsyncMock(return_value=TaskResult(
            task_id="task_1",
            output="print('hello')",
            score=0.9,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
        ))
        
        # Act
        state = await orch.run_project(
            project_description="Test project",
            success_criteria="Works",
        )
        
        # Assert
        assert state.status.value in ["completed", "partial_success"]
        assert budget._reserved_usd == 0, f"Reservation leaked after run_project: {budget._reserved_usd}"

    @pytest.mark.asyncio
    async def test_reserve_with_cli_entry_point(self):
        """
        TEST-ID: BUG-CONC-001-INT-02
        
        DESCRIPTION: Verify reserve pattern works through CLI entry point.
        
        PRECONDITIONS:
        - CLI with --budget flag
        - Valid project spec
        
        INPUT:
        - CLI command: orchestrator --project "test" --budget 1.00
        
        EXPECTED BEHAVIOR (with patch):
        - CLI executes project
        - Budget managed correctly
        - Clean exit with no reservation leaks
        
        FAILURE BEHAVIOR (without patch):
        - CLI may hang or crash on budget overcommitment
        """
        # This is a placeholder for CLI integration test
        # Full implementation would require subprocess call to CLI
        # For now, verify the Budget class is importable and usable
        from orchestrator.cli import main
        
        budget = Budget(max_usd=1.00)
        assert budget.reserve(0.50) is True
        assert budget._reserved_usd == 0.50
        budget.release_reservation(0.50)
        assert budget._reserved_usd == 0
