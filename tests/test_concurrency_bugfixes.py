"""
Regression Tests for Concurrency Bug Fixes
===========================================
Test IDs: BUG-CONC-001, BUG-BUDGET-004, BUG-ROUTER-003

These tests verify the fixes for critical concurrency bugs in the
Budget and AdaptiveRouter classes.

Test Framework: pytest with pytest-asyncio
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.models import Budget
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-CONC-001: Budget Lock Initialization Race Condition
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetEagerLockInitialization:
    """
    Regression tests for BUG-CONC-001:
    Lazy lock initialization created race condition when multiple
    concurrent tasks called Budget methods simultaneously.
    
    Fix: Lock is now eagerly initialized in __post_init__.
    """
    
    def test_lock_eagerly_initialized(self):
        """REGRESSION-001: Verify lock is initialized synchronously."""
        budget = Budget(max_usd=100.0)
        
        # Lock should be initialized immediately, not lazily
        assert budget._lock is not None, "Lock must be initialized in __init__"
        assert isinstance(budget._lock, asyncio.Lock), "Lock must be asyncio.Lock"
    
    def test_lock_type_correct(self):
        """REGRESSION-002: Verify lock is correct type for async operations."""
        budget = Budget(max_usd=50.0)
        
        # Must be asyncio.Lock, not threading.Lock
        assert type(budget._lock).__name__ == "Lock"
        assert hasattr(budget._lock, "acquire")
        assert hasattr(budget._lock, "release")
    
    @pytest.mark.asyncio
    async def test_concurrent_charges_no_race(self):
        """
        REGRESSION-003: Main regression test for BUG-CONC-001.
        
        Without fix: Multiple tasks could create separate locks,
        allowing race conditions in budget updates.
        
        With fix: All tasks share same lock, charges are serialized.
        """
        budget = Budget(max_usd=1000.0)
        num_tasks = 20
        charge_per_task = 5.0
        
        async def charge_worker():
            await budget.charge(charge_per_task, "generation")
        
        # Run all tasks concurrently
        await asyncio.gather(*[charge_worker() for _ in range(num_tasks)])
        
        # Verify exact total (no lost updates)
        expected = num_tasks * charge_per_task
        assert budget.spent_usd == expected, (
            f"Race condition detected: expected ${expected}, got ${budget.spent_usd}"
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_reserves_no_race(self):
        """REGRESSION-004: Concurrent reservations should be atomic."""
        budget = Budget(max_usd=100.0)
        successful_reserves = []
        
        async def try_reserve(amount, worker_id):
            result = await budget.reserve(amount)
            if result:
                successful_reserves.append(worker_id)
        
        # 10 workers try to reserve $15 each (only 6 should succeed)
        await asyncio.gather(*[
            try_reserve(15.0, i) for i in range(10)
        ])
        
        # Exactly 6 should succeed (6 * 15 = 90 <= 100)
        assert len(successful_reserves) == 6
        assert budget._reserved_usd == 90.0


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-BUDGET-004: Budget Reservation Leakage on Charge Failure
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetReservationFailureRecovery:
    """
    Regression tests for BUG-BUDGET-004:
    If charge() failed after reservation was released, budget was permanently
    incorrect (reserved amount lost).
    
    Fix: commit_reservation() now restores reservation if charge() fails.
    """
    
    @pytest.mark.asyncio
    async def test_commit_restores_on_charge_failure(self):
        """
        REGRESSION-005: Main regression test for BUG-BUDGET-004.
        
        Without fix: charge() failure leaves _reserved_usd = 0 but
        spent_usd unchanged, causing budget discrepancy.
        
        With fix: _reserved_usd is restored when charge() fails.
        """
        budget = Budget(max_usd=100.0)
        
        # Reserve $10
        reserved = await budget.reserve(10.0)
        assert reserved
        assert budget._reserved_usd == 10.0
        
        # Mock charge to fail
        with patch.object(budget, 'charge', new_callable=AsyncMock) as mock_charge:
            mock_charge.side_effect = Exception("Simulated failure")
            
            # Try to commit - should fail and restore
            with pytest.raises(Exception):
                await budget.commit_reservation(8.0)
        
        # Verify reservation was restored
        assert budget._reserved_usd == 10.0, (
            f"Budget leakage: expected $10 reserved, got ${budget._reserved_usd}"
        )
    
    @pytest.mark.asyncio
    async def test_commit_succeeds_normal_case(self):
        """REGRESSION-006: Normal commit (no failure) should work correctly."""
        budget = Budget(max_usd=100.0)
        
        # Reserve $10
        await budget.reserve(10.0)
        assert budget._reserved_usd == 10.0
        
        # Commit $8
        await budget.commit_reservation(8.0, "generation")
        
        # Verify: reservation released, amount charged
        assert budget._reserved_usd == 0.0
        assert budget.spent_usd == 8.0
    
    @pytest.mark.asyncio
    async def test_multiple_reserve_commit_cycles(self):
        """REGRESSION-007: Multiple reserve/commit cycles should not leak."""
        budget = Budget(max_usd=1000.0)
        
        for i in range(10):
            # Reserve
            await budget.reserve(20.0)
            # Commit
            await budget.commit_reservation(18.0, "generation")
        
        # Should have exactly 10 * 18 = 180 spent, 0 reserved
        assert budget.spent_usd == 180.0
        assert budget._reserved_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-ROUTER-003: AdaptiveRouter Data Race on Model Availability Check
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouterAsyncSafeAvailability:
    """
    Regression tests for BUG-ROUTER-003:
    is_available() read shared state without lock protection,
    causing data races with concurrent writers.
    
    Fix: Added is_available_async() for async-safe reads.
    """
    
    def test_sync_is_available_basic(self):
        """REGRESSION-008: Basic sync is_available() functionality."""
        router = AdaptiveRouter()
        
        # All models should be healthy initially
        assert router.is_available(Model.GPT_4O) == True
        assert router.is_available(Model.GPT_4O_MINI) == True
    
    @pytest.mark.asyncio
    async def test_async_is_available_basic(self):
        """REGRESSION-009: New async-safe is_available_async()."""
        router = AdaptiveRouter()
        
        # All models should be healthy initially
        assert await router.is_available_async(Model.GPT_4O) == True
        assert await router.is_available_async(Model.GPT_4O_MINI) == True
    
    @pytest.mark.asyncio
    async def test_is_available_after_timeout(self):
        """REGRESSION-010: Availability changes after timeouts."""
        router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=60.0)
        
        # Record 3 timeouts (threshold)
        for _ in range(3):
            await router.record_timeout(Model.GPT_4O)
        
        # Should be degraded (unavailable)
        assert await router.is_available_async(Model.GPT_4O) == False
        assert router.is_available(Model.GPT_4O) == False
        
        # Other models should still be available
        assert await router.is_available_async(Model.GPT_4O_MINI) == True
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write_no_crash(self):
        """
        REGRESSION-011: Concurrent reads and writes should not crash.
        
        This is the main regression test for BUG-ROUTER-003.
        """
        router = AdaptiveRouter(timeout_threshold=5, cooldown_seconds=30.0)
        model = Model.GPT_4O
        errors = []
        
        async def writer():
            for _ in range(50):
                await router.record_timeout(model)
                await asyncio.sleep(0.001)
        
        async def reader():
            for _ in range(50):
                try:
                    # Test both sync and async versions
                    router.is_available(model)
                    await router.is_available_async(model)
                except Exception as e:
                    errors.append(e)
                await asyncio.sleep(0.001)
        
        # Run concurrently
        await asyncio.gather(writer(), reader())
        
        # Should have no errors
        assert len(errors) == 0, f"Data race errors: {errors}"
    
    @pytest.mark.asyncio
    async def test_auth_failure_isolation(self):
        """REGRESSION-012: Auth failure on one model doesn't affect others."""
        router = AdaptiveRouter()
        
        # Disable one model
        await router.record_auth_failure(Model.GPT_4O)
        
        # Check isolation
        assert await router.is_available_async(Model.GPT_4O) == False
        assert await router.is_available_async(Model.GPT_4O_MINI) == True
        assert await router.is_available_async(Model.GEMINI_FLASH) == True


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetRouterIntegration:
    """
    Integration tests verifying Budget and AdaptiveRouter work together
    correctly under concurrent load.
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_routing_with_budget(self):
        """
        REGRESSION-013: Integration test - routing decisions + budget charges.
        
        Simulates real-world scenario where multiple tasks:
        1. Check model availability
        2. Reserve budget
        3. Execute (simulate)
        4. Commit budget
        """
        budget = Budget(max_usd=100.0)
        router = AdaptiveRouter()
        
        async def worker(task_id: int):
            # Check model availability
            if not await router.is_available_async(Model.GPT_4O):
                return f"Task {task_id}: Model unavailable"
            
            # Reserve budget
            if not await budget.reserve(5.0):
                return f"Task {task_id}: Budget unavailable"
            
            # Simulate work
            await asyncio.sleep(0.001)
            
            # Commit
            await budget.commit_reservation(4.5, "generation")
            return f"Task {task_id}: Success"
        
        # Run 15 concurrent workers
        results = await asyncio.gather(*[worker(i) for i in range(15)])
        
        # All should succeed
        successes = [r for r in results if "Success" in r]
        assert len(successes) == 15
        
        # Budget should be correct
        assert budget.spent_usd == 15 * 4.5
        assert budget._reserved_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case tests for robustness."""
    
    @pytest.mark.asyncio
    async def test_budget_zero_charge(self):
        """REGRESSION-014: Zero charge should not cause issues."""
        budget = Budget(max_usd=100.0)
        await budget.charge(0.0, "generation")
        assert budget.spent_usd == 0.0
    
    @pytest.mark.asyncio
    async def test_budget_negative_reserve_rejected(self):
        """REGRESSION-015: Negative reserve should raise ValueError."""
        budget = Budget(max_usd=100.0)
        
        with pytest.raises(ValueError):
            await budget.reserve(-5.0)
    
    @pytest.mark.asyncio
    async def test_router_preferred_model_empty_candidates(self):
        """REGRESSION-016: preferred_model with empty candidates."""
        router = AdaptiveRouter()
        result = await router.preferred_model([])
        assert result is None
    
    @pytest.mark.asyncio
    async def test_router_all_models_degraded(self):
        """REGRESSION-017: preferred_model when all models degraded."""
        router = AdaptiveRouter(timeout_threshold=1, cooldown_seconds=3600.0)
        
        # Degrade all candidates
        for model in [Model.GPT_4O, Model.GPT_4O_MINI]:
            await router.record_timeout(model)
        
        # Should return None (no healthy models)
        result = await router.preferred_model([Model.GPT_4O, Model.GPT_4O_MINI])
        assert result is None
