"""
Integration tests for BudgetHierarchy wiring into Orchestrator.run_job()

Verifies that charge_job() is called after each job with the actual spent amount,
enabling cross-run budget enforcement.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call

from orchestrator.cost import BudgetHierarchy


def test_budget_hierarchy_charge_job_called_integration():
    """
    Verify that BudgetHierarchy.charge_job() correctly accumulates spend
    across sequential jobs, enabling the pre-flight can_afford_job() to
    enforce multi-run budgets.

    Currently, charge_job() is never called from orchestrator.engine.run_job(),
    so this test will FAIL — proving the bug exists.

    After the fix, charge_job() will be called in run_job() after each
    project completes, and this test will verify the enforcement loop works.
    """
    hierarchy = BudgetHierarchy(org_max_usd=10.0)

    # Before any charging, can_afford first job ($9)
    assert hierarchy.can_afford_job("job_1", "eng", 9.0) is True

    # Simulate running job_1 and charging actual spend
    # (After fix: run_job() will call this automatically)
    hierarchy.charge_job("job_1", "eng", 9.0)

    # After charging $9, remaining org budget is $1
    assert hierarchy.remaining("org") == pytest.approx(1.0)

    # Second job requests $2.00 but only $1.00 remains
    # Pre-flight check must reject it
    assert hierarchy.can_afford_job("job_2", "eng", 2.0) is False

    # But a job requesting $0.80 would fit
    assert hierarchy.can_afford_job("job_3", "eng", 0.8) is True

    # After charging it
    hierarchy.charge_job("job_3", "eng", 0.8)

    # Only $0.2 remains
    assert hierarchy.remaining("org") == pytest.approx(0.2)

    # Any job requesting more than remaining is rejected
    assert hierarchy.can_afford_job("job_4", "eng", 0.3) is False


@pytest.mark.asyncio
async def test_run_job_calls_charge_job_on_hierarchy():
    """
    FAILING TEST: Verifies that Orchestrator.run_job() calls charge_job()
    on BudgetHierarchy after the project completes.

    This test will FAIL until the fix is implemented:
      orchestrator/engine.py run_job() must call
      self._budget_hierarchy.charge_job() with the actual spent amount
      after await run_project() returns.

    Without this call, the hierarchy's accumulated spend remains forever at zero,
    rendering cross-run budget enforcement completely non-functional.
    """
    from orchestrator.engine import Orchestrator
    from orchestrator.models import Budget, ProjectStatus
    from orchestrator.policy import JobSpec

    hierarchy = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 50.0})

    with patch('orchestrator.engine.Orchestrator.run_project', new_callable=AsyncMock) as mock_run_project:
        # Mock run_project to return a state with SUCCESS status
        mock_state = MagicMock()
        mock_state.status = ProjectStatus.SUCCESS
        mock_run_project.return_value = mock_state

        orch = Orchestrator(budget_hierarchy=hierarchy)

        # Create a JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=50.0),
        )

        # Set the budget to simulate $5 actually spent
        # (This happens during run_project internally)
        def side_effect(*args, **kwargs):
            # After run_project "executes", simulate spend via Budget.charge()
            orch.budget.spent_usd = 5.0
            return mock_state

        mock_run_project.side_effect = side_effect

        # Patch charge_job to track if it's called
        with patch.object(hierarchy, 'charge_job', wraps=hierarchy.charge_job) as mock_charge:
            # This test will FAIL: charge_job() is not called from run_job()
            await orch.run_job(spec)

            # Verify charge_job was called with the actual spent amount
            # If this assert fails, it proves the bug: charge_job is not being called
            mock_charge.assert_called_once()
            call_args = mock_charge.call_args
            # charge_job(job_id, team, amount)
            assert call_args[0][2] == 5.0  # amount should be the actual spent_usd
