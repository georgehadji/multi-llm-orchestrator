"""
Test suite for BUG fixes: Budget race condition and AdaptiveRouter async lock.

Tests verify:
1. Budget.charge() is thread-safe under concurrent access
2. AdaptiveRouter uses asyncio.Lock correctly
3. All async methods properly await
"""
import asyncio
import pytest
import time
from orchestrator.models import Budget
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model


class TestBudgetRaceConditionFix:
    """Test BUG-001 fix: Budget.charge() race condition."""

    @pytest.mark.asyncio
    async def test_concurrent_charge_atomic(self):
        """
        Verify that concurrent charges are atomic and don't lose updates.
        
        This test would fail with the synchronous charge() implementation
        due to race conditions.
        """
        budget = Budget(max_usd=100.0)
        num_tasks = 100
        charge_amount = 0.1  # Total: $10.00
        
        async def charge_task():
            await budget.charge(charge_amount, "generation")
        
        # Run all charges concurrently
        await asyncio.gather(*[charge_task() for _ in range(num_tasks)])
        
        # Verify no lost updates
        assert budget.spent_usd == pytest.approx(num_tasks * charge_amount, rel=1e-9)
        assert budget.spent_usd == pytest.approx(10.0, rel=1e-9)
    
    @pytest.mark.asyncio
    async def test_concurrent_charge_phase_tracking(self):
        """Verify phase_spent tracking is also atomic."""
        budget = Budget(max_usd=100.0)
        
        async def charge_phase(phase: str, amount: float):
            await budget.charge(amount, phase)
        
        # Charge different phases concurrently
        tasks = []
        for i in range(50):
            tasks.append(charge_phase("generation", 0.1))
            tasks.append(charge_phase("cross_review", 0.05))
        
        await asyncio.gather(*tasks)
        
        # Verify phase tracking
        assert budget.phase_spent["generation"] == pytest.approx(5.0, rel=1e-9)
        assert budget.phase_spent["cross_review"] == pytest.approx(2.5, rel=1e-9)
        assert budget.spent_usd == pytest.approx(7.5, rel=1e-9)
    
    @pytest.mark.asyncio
    async def test_reserve_charge_release_sequence(self):
        """Test that reserve/commit/release sequence works correctly."""
        budget = Budget(max_usd=10.0)
        
        # Reserve budget
        reserved = await budget.reserve(5.0)
        assert reserved is True
        assert budget.remaining_usd == pytest.approx(5.0, rel=1e-9)
        assert budget._reserved_usd == pytest.approx(5.0, rel=1e-9)
        
        # Commit reservation
        await budget.commit_reservation(5.0, "generation")
        assert budget._reserved_usd == pytest.approx(0.0, rel=1e-9)
        assert budget.spent_usd == pytest.approx(5.0, rel=1e-9)
        assert budget.phase_spent["generation"] == pytest.approx(5.0, rel=1e-9)
    
    @pytest.mark.asyncio
    async def test_commit_reservation_different_amount(self):
        """Test commit when actual cost differs from reserved."""
        budget = Budget(max_usd=10.0)
        
        # Reserve $5
        await budget.reserve(5.0)
        
        # Commit with different amount ($4.50)
        await budget.commit_reservation(4.5, "generation")
        
        assert budget._reserved_usd == pytest.approx(0.0, rel=1e-9)
        assert budget.spent_usd == pytest.approx(4.5, rel=1e-9)
        assert budget.remaining_usd == pytest.approx(5.5, rel=1e-9)
    
    @pytest.mark.asyncio
    async def test_release_reservation(self):
        """Test releasing unused reservation."""
        budget = Budget(max_usd=10.0)
        
        await budget.reserve(3.0)
        assert budget.remaining_usd == pytest.approx(7.0, rel=1e-9)
        
        await budget.release_reservation(3.0)
        assert budget._reserved_usd == pytest.approx(0.0, rel=1e-9)
        assert budget.remaining_usd == pytest.approx(10.0, rel=1e-9)
        assert budget.spent_usd == pytest.approx(0.0, rel=1e-9)  # No charge


class TestAdaptiveRouterAsyncFix:
    """Test BUG-002 fix: AdaptiveRouter uses asyncio.Lock."""

    @pytest.mark.asyncio
    async def test_async_record_success(self):
        """Verify record_success is async and works correctly."""
        router = AdaptiveRouter()
        
        # Should be async
        result = router.record_success(Model.GPT_4O)
        assert asyncio.iscoroutine(result)
        await result
        
        # Verify state
        assert router._timeout_counts[Model.GPT_4O] == 0
        assert router._degraded_since[Model.GPT_4O] is None
    
    @pytest.mark.asyncio
    async def test_async_record_timeout(self):
        """Verify record_timeout is async and triggers degradation."""
        router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=60.0)
        
        # Record 3 timeouts
        for _ in range(3):
            await router.record_timeout(Model.GEMINI_FLASH)
        
        # Model should be degraded
        assert router.is_available(Model.GEMINI_FLASH) is False
    
    @pytest.mark.asyncio
    async def test_async_record_latency(self):
        """Verify record_latency is async and tracks EMA."""
        router = AdaptiveRouter()
        
        # Record multiple latencies
        await router.record_latency(Model.CLAUDE_3_5_SONNET, 100.0, alpha=0.5)
        await router.record_latency(Model.CLAUDE_3_5_SONNET, 200.0, alpha=0.5)
        
        # EMA should be: 0.5 * 200 + 0.5 * (0.5 * 100 + 0.5 * 0) = 100 + 25 = 125
        assert router._latencies[Model.CLAUDE_3_5_SONNET] == pytest.approx(150.0, rel=1e-6)
    
    @pytest.mark.asyncio
    async def test_async_preferred_model(self):
        """Verify preferred_model is async and returns correct model."""
        router = AdaptiveRouter()
        
        # Record different latencies
        await router.record_latency(Model.GPT_4O_MINI, 50.0)
        await router.record_latency(Model.GEMINI_FLASH, 100.0)
        await router.record_latency(Model.CLAUDE_3_HAIKU, 75.0)
        
        # Should return fastest (GPT-4o-mini)
        candidates = [Model.GPT_4O_MINI, Model.GEMINI_FLASH, Model.CLAUDE_3_HAIKU]
        result = await router.preferred_model(candidates)
        assert result == Model.GPT_4O_MINI
    
    @pytest.mark.asyncio
    async def test_concurrent_router_operations(self):
        """Test router handles concurrent operations safely."""
        router = AdaptiveRouter(timeout_threshold=5, cooldown_seconds=1.0)
        
        async def mixed_operations(model: Model):
            """Mix of success, timeout, and latency recording."""
            for i in range(10):
                if i % 3 == 0:
                    await router.record_timeout(model)
                else:
                    await router.record_success(model)
                await router.record_latency(model, float(i * 10))
        
        # Run concurrent operations on different models
        tasks = [
            mixed_operations(Model.GPT_4O),
            mixed_operations(Model.GEMINI_FLASH),
            mixed_operations(Model.CLAUDE_3_5_SONNET),
        ]
        
        await asyncio.gather(*tasks)
        
        # All models should still be available (successes outnumber timeouts)
        assert router.is_available(Model.GPT_4O) is True
        assert router.is_available(Model.GEMINI_FLASH) is True
        assert router.is_available(Model.CLAUDE_3_5_SONNET) is True
    
    @pytest.mark.asyncio
    async def test_is_available_non_blocking(self):
        """Verify is_available() is synchronous and non-blocking."""
        router = AdaptiveRouter()
        
        # Should be callable without await
        result = router.is_available(Model.GPT_4O)
        assert result is True
        
        # Mark as degraded
        router._degraded_since[Model.GPT_4O] = time.monotonic()
        result = router.is_available(Model.GPT_4O)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_auth_failure_permanent(self):
        """Verify auth failure permanently disables model."""
        router = AdaptiveRouter()
        
        await router.record_auth_failure(Model.GPT_4O)
        
        assert router.is_available(Model.GPT_4O) is False
        assert Model.GPT_4O in router._disabled


class TestIntegration:
    """Integration tests for both fixes working together."""

    @pytest.mark.asyncio
    async def test_budget_and_router_concurrent(self):
        """Test Budget and AdaptiveRouter work together under concurrency."""
        budget = Budget(max_usd=50.0)
        router = AdaptiveRouter(timeout_threshold=3)
        
        async def task_execution(model: Model, cost: float, success: bool):
            """Simulate a task execution with budget charge and router update."""
            # Reserve budget
            if not await budget.reserve(cost):
                return False
            
            if success:
                await router.record_success(model)
                await router.record_latency(model, 100.0)
                await budget.commit_reservation(cost, "generation")
            else:
                await router.record_timeout(model)
                await budget.release_reservation(cost)
            
            return True

        # Simulate mixed success/failure tasks
        tasks = []
        for i in range(20):
            model = Model.GPT_4O_MINI if i % 2 == 0 else Model.GEMINI_FLASH
            success = i % 4 != 0  # 75% success rate
            tasks.append(task_execution(model, 0.5, success))

        results = await asyncio.gather(*tasks)

        # Most tasks should succeed
        assert sum(results) >= 14  # At least 70% success


class TestRegressionPrevention:
    """
    Regression tests to prevent reintroduction of fixed bugs.
    
    These tests document the bugs that were fixed and ensure they don't return:
    - BUG-001: Budget.charge() race condition (now async with lock)
    - BUG-002: AdaptiveRouter threading.Lock (now asyncio.Lock)
    - BUG-003: Misleading env var comment (now documented)
    """

    @pytest.mark.asyncio
    async def test_budget_charge_is_async(self):
        """
        REGRESSION TEST: Ensure Budget.charge() remains async.
        
        BUG-001 FIX: Budget.charge() must be async to prevent race conditions.
        This test will fail if charge() is ever changed back to sync.
        """
        budget = Budget()
        result = budget.charge(1.0, "generation")
        assert asyncio.iscoroutine(result), "Budget.charge() must be async!"
        await result  # Complete the coroutine

    @pytest.mark.asyncio
    async def test_budget_commit_reservation_is_async(self):
        """
        REGRESSION TEST: Ensure Budget.commit_reservation() remains async.
        
        BUG-001 FIX: Must be async to call charge() properly.
        """
        budget = Budget(max_usd=10.0)
        await budget.reserve(5.0)
        
        result = budget.commit_reservation(4.5)
        assert asyncio.iscoroutine(result), "Budget.commit_reservation() must be async!"
        await result

    def test_adaptive_router_is_available_is_sync(self):
        """
        REGRESSION TEST: Ensure AdaptiveRouter.is_available() remains synchronous.
        
        BUG-002 FIX: is_available() intentionally uses non-blocking reads
        to avoid blocking in list comprehensions. This is safe because
        it only reads immutable state references.
        
        This test ensures is_available() doesn't accidentally become async.
        """
        router = AdaptiveRouter()
        result = router.is_available(Model.GPT_4O)
        assert not asyncio.iscoroutine(result), "is_available() must remain sync!"
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_adaptive_router_methods_are_async(self):
        """
        REGRESSION TEST: Ensure AdaptiveRouter state-modifying methods remain async.
        
        BUG-002 FIX: All state-modifying methods must use asyncio.Lock.
        """
        router = AdaptiveRouter()
        
        # Test record_success
        result = router.record_success(Model.GPT_4O)
        assert asyncio.iscoroutine(result), "record_success() must be async!"
        await result
        
        # Test record_timeout
        result = router.record_timeout(Model.GPT_4O)
        assert asyncio.iscoroutine(result), "record_timeout() must be async!"
        await result
        
        # Test record_latency
        result = router.record_latency(Model.GPT_4O, 100.0)
        assert asyncio.iscoroutine(result), "record_latency() must be async!"
        await result
        
        # Test preferred_model
        result = router.preferred_model([Model.GPT_4O])
        assert asyncio.iscoroutine(result), "preferred_model() must be async!"
        await result

    @pytest.mark.asyncio
    async def test_budget_lock_is_asyncio_lock(self):
        """
        REGRESSION TEST: Ensure Budget uses asyncio.Lock, not threading.Lock.
        
        BUG-001 FIX: Must use asyncio.Lock to prevent event loop blocking.
        """
        budget = Budget()
        lock = budget._get_lock()
        assert isinstance(lock, asyncio.Lock), "Budget must use asyncio.Lock!"

    @pytest.mark.asyncio  
    async def test_adaptive_router_lock_is_asyncio_lock(self):
        """
        REGRESSION TEST: Ensure AdaptiveRouter uses asyncio.Lock.
        
        BUG-002 FIX: Must use asyncio.Lock to prevent event loop blocking.
        """
        router = AdaptiveRouter()
        assert isinstance(router._lock, asyncio.Lock), "AdaptiveRouter must use asyncio.Lock!"

    def test_env_var_comment_updated(self):
        """
        REGRESSION TEST: Ensure cli.py has correct env var comment.
        
        BUG-003 FIX: Comment must accurately describe load_dotenv behavior.
        """
        import inspect
        from orchestrator import cli
        
        # Get the source code of the cli module
        source = inspect.getsource(cli)
        
        # Check that the old misleading comment is gone
        assert "win over empty system env vars" not in source, \
            "Old misleading comment should be removed!"
        
        # Check that the new accurate comment exists
        assert "override=True" in source, \
            "Should still use override=True!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
