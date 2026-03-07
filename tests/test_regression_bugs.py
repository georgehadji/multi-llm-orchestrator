"""
Regression tests for BUG-001: TestFixer import failures
"""
import pytest
import asyncio


class TestTestFixerImports:
    """Test that TestFixer module can be imported without errors."""

    def test_test_fixer_import(self):
        """Test that TestFixer can be imported."""
        from orchestrator.test_fixer import TestFixer
        assert TestFixer is not None

    def test_budget_manager_import(self):
        """Test that BudgetManager can be imported."""
        from orchestrator.test_fixer import BudgetManager
        assert BudgetManager is not None

    def test_budget_manager_instantiation(self):
        """Test that BudgetManager can be instantiated."""
        from orchestrator.test_fixer import BudgetManager
        bm = BudgetManager(max_cost=1.0, max_time=60)
        assert bm is not None
        assert bm.is_exhausted() == False

    def test_budget_manager_exhausted(self):
        """Test that BudgetManager correctly reports exhaustion."""
        from orchestrator.test_fixer import BudgetManager
        bm = BudgetManager(max_cost=0.0, max_time=0)  # Zero budget
        assert bm.is_exhausted() == True

    def test_test_fixer_instantiation(self):
        """Test that TestFixer can be instantiated."""
        from orchestrator.test_fixer import TestFixer
        fixer = TestFixer()
        assert fixer is not None
        assert fixer._budget is not None


class TestCacheImports:
    """Test that cache module works correctly."""

    def test_cache_import(self):
        """Test that DiskCache can be imported."""
        from orchestrator.cache import DiskCache
        assert DiskCache is not None

    def test_cache_instantiation(self):
        """Test that DiskCache can be instantiated."""
        from orchestrator.cache import DiskCache
        cache = DiskCache()
        assert cache is not None

    @pytest.mark.asyncio
    async def test_cache_connection(self):
        """Test that cache connection can be obtained."""
        from orchestrator.cache import DiskCache
        cache = DiskCache()
        conn = await cache._get_conn()
        assert conn is not None

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        """Test that cache handles concurrent access without race conditions."""
        from orchestrator.cache import DiskCache
        
        cache = DiskCache()
        
        async def get_conn():
            return await cache._get_conn()
        
        # Run 5 concurrent connections
        tasks = [get_conn() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should get a valid connection
        assert all(r is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])