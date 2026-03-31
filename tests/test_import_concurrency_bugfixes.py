"""
Regression Tests for Import and Concurrency Bug Fixes
======================================================
Test IDs: BUG-API-001, BUG-CACHE-002, BUG-STATE-003

These tests verify the fixes for critical initialization and concurrency bugs
in the UnifiedClient, DiskCache, and StateManager classes.

Test Framework: pytest with pytest-asyncio
"""

import asyncio
import os
import pytest

from orchestrator.api_clients import UnifiedClient
from orchestrator.cache import DiskCache
from orchestrator.state import StateManager

# ═══════════════════════════════════════════════════════════════════════════════
# BUG-API-001: Missing os Import in api_clients.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifiedClientImportFix:
    """
    Regression tests for BUG-API-001:
    UnifiedClient.__init__() referenced os.environ before os was imported.

    Fix: Added 'import os' at module level (line 25).
    """

    def test_unified_client_init_no_nameerror(self):
        """
        REGRESSION-001: Main regression test for BUG-API-001.

        Without fix: NameError: name 'os' is not defined

        With fix: Client initializes successfully.
        """
        # This would have raised NameError before the fix
        client = UnifiedClient()

        assert client is not None
        assert hasattr(client, "xai_region")
        assert hasattr(client, "semaphore")
        assert hasattr(client, "cache")

    def test_unified_client_xai_region_default(self):
        """REGRESSION-002: XAI_REGION defaults to None when not set."""
        # Ensure XAI_REGION is not set
        original = os.environ.pop("XAI_REGION", None)

        try:
            client = UnifiedClient()
            assert client.xai_region is None
        finally:
            # Restore original value
            if original is not None:
                os.environ["XAI_REGION"] = original

    def test_unified_client_xai_region_from_env(self):
        """REGRESSION-003: XAI_REGION correctly read from environment."""
        # Set XAI_REGION
        os.environ["XAI_REGION"] = "eu-west-1"

        try:
            client = UnifiedClient()
            assert client.xai_region == "eu-west-1"
        finally:
            del os.environ["XAI_REGION"]

    def test_unified_client_xai_region_explicit_param(self):
        """REGRESSION-004: Explicit xai_region parameter takes precedence."""
        os.environ["XAI_REGION"] = "eu-west-1"

        try:
            # Explicit parameter should override env var
            client = UnifiedClient(xai_region="us-east-1")
            assert client.xai_region == "us-east-1"
        finally:
            del os.environ["XAI_REGION"]


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-CACHE-002: DiskCache Lazy Lock Initialization Race Condition
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiskCacheEagerLockFix:
    """
    Regression tests for BUG-CACHE-002:
    DiskCache._lock was lazily initialized, creating race condition when
    multiple concurrent calls accessed the cache simultaneously.

    Fix: Lock is now eagerly initialized in __init__.
    """

    def test_lock_eagerly_initialized(self):
        """REGRESSION-005: Verify lock is initialized synchronously."""
        cache = DiskCache()

        # Lock should be initialized immediately, not lazily
        assert cache._lock is not None, "Lock must be initialized in __init__"
        assert isinstance(cache._lock, asyncio.Lock), "Lock must be asyncio.Lock"

    def test_lock_identity_constant(self):
        """REGRESSION-006: Lock identity must remain constant."""
        cache = DiskCache()
        lock_id_1 = id(cache._lock)

        # Access connection multiple times
        asyncio.run(cache._get_conn())
        lock_id_2 = id(cache._lock)

        assert lock_id_1 == lock_id_2, "Lock identity must remain constant"

    @pytest.mark.asyncio
    async def test_concurrent_get_put_no_race(self):
        """
        REGRESSION-007: Main regression test for BUG-CACHE-002.

        Without fix: Multiple tasks could create separate locks,
        allowing race conditions in cache operations.

        With fix: All operations serialized correctly.
        """
        cache = DiskCache()

        async def worker(i: int):
            key = f"test_key_{i}"
            await cache.put("test-model", key, 100, f"value_{i}")
            result = await cache.get("test-model", key, 100)
            return result is not None

        # Run 20 concurrent operations
        tasks = [worker(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        assert success_count == 20, f"Expected 20 successes, got {success_count}"

    @pytest.mark.asyncio
    async def test_concurrent_connection_sharing(self):
        """REGRESSION-008: Concurrent _get_conn() calls share same connection."""
        cache = DiskCache()
        connections = []

        async def get_conn(i: int):
            conn = await cache._get_conn()
            connections.append(conn)

        # Run 10 concurrent connection requests
        await asyncio.gather(*[get_conn(i) for i in range(10)])

        # All should get the same connection (singleton pattern)
        first_conn = connections[0]
        same_conn_count = sum(1 for c in connections if c is first_conn)
        assert same_conn_count == 10, "All should share same connection"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-STATE-003: StateManager Lazy Lock Initialization Race Condition
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateManagerEagerLockFix:
    """
    Regression tests for BUG-STATE-003:
    StateManager._lock was lazily initialized, creating identical race condition
    as BUG-CACHE-002.

    Fix: Lock is now eagerly initialized in __init__.
    """

    def test_lock_eagerly_initialized(self):
        """REGRESSION-009: Verify lock is initialized synchronously."""
        state_mgr = StateManager()

        # Lock should be initialized immediately, not lazily
        assert state_mgr._lock is not None, "Lock must be initialized in __init__"
        assert isinstance(state_mgr._lock, asyncio.Lock), "Lock must be asyncio.Lock"

    def test_lock_identity_constant(self):
        """REGRESSION-010: Lock identity must remain constant."""
        state_mgr = StateManager()
        lock_id_1 = id(state_mgr._lock)

        # Access connection (this would trigger lazy init before fix)
        asyncio.run(state_mgr._get_conn())
        lock_id_2 = id(state_mgr._lock)

        assert lock_id_1 == lock_id_2, "Lock identity must remain constant"

    @pytest.mark.asyncio
    async def test_concurrent_connection_no_race(self):
        """
        REGRESSION-011: Main regression test for BUG-STATE-003.

        Without fix: Multiple tasks could create separate locks,
        allowing race conditions in state operations.

        With fix: All operations serialized correctly.
        """
        state_mgr = StateManager()
        connections = []
        errors = []

        async def get_conn(i: int):
            try:
                conn = await state_mgr._get_conn()
                connections.append((i, conn))
            except Exception as e:
                errors.append((i, e))

        # Run 10 concurrent connection requests
        await asyncio.gather(*[get_conn(i) for i in range(10)])

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All should get the same connection
        if connections:
            first_conn = connections[0][1]
            same_conn_count = sum(1 for _, c in connections if c is first_conn)
            assert same_conn_count == 10, "All should share same connection"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationFixes:
    """
    Integration tests verifying all fixes work together correctly.
    """

    def test_client_with_cache_and_state(self):
        """
        REGRESSION-012: Integration test - UnifiedClient + DiskCache + StateManager.

        Verifies all three fixes work together without conflicts.
        """
        # Create all components
        cache = DiskCache()
        state_mgr = StateManager()
        client = UnifiedClient(cache=cache)

        # Verify all locks are initialized
        assert isinstance(cache._lock, asyncio.Lock)
        assert isinstance(state_mgr._lock, asyncio.Lock)
        assert isinstance(client.cache._lock, asyncio.Lock)

        # Verify client can access os module
        assert hasattr(client, "xai_region")

    @pytest.mark.asyncio
    async def test_concurrent_client_cache_operations(self):
        """
        REGRESSION-013: Concurrent client calls with shared cache.

        Simulates real-world scenario where multiple API calls
        share the same cache instance.
        """
        cache = DiskCache()
        client = UnifiedClient(cache=cache)

        # Note: We can't actually call client.call() without API keys,
        # but we can test cache operations that the client would use
        async def cache_worker(i: int):
            key = f"client_key_{i}"
            await cache.put("gpt-4o", key, 100, f"response_{i}")
            result = await cache.get("gpt-4o", key, 100)
            return result is not None

        # Run 15 concurrent cache operations
        tasks = [cache_worker(i) for i in range(15)]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        assert success_count == 15, f"Expected 15 successes, got {success_count}"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_cache_multiple_instances(self):
        """REGRESSION-014: Multiple DiskCache instances each have own lock."""
        cache1 = DiskCache()
        cache2 = DiskCache()

        assert cache1._lock is not cache2._lock
        assert isinstance(cache1._lock, asyncio.Lock)
        assert isinstance(cache2._lock, asyncio.Lock)

    def test_state_multiple_instances(self):
        """REGRESSION-015: Multiple StateManager instances each have own lock."""
        state1 = StateManager()
        state2 = StateManager()

        assert state1._lock is not state2._lock
        assert isinstance(state1._lock, asyncio.Lock)
        assert isinstance(state2._lock, asyncio.Lock)

    def test_client_no_api_keys(self):
        """REGRESSION-016: UnifiedClient initializes even without API keys."""
        # Save original env vars
        original_keys = {}
        for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                original_keys[key] = os.environ.pop(key)

        try:
            # Should not raise, even without any API keys
            client = UnifiedClient()
            assert client is not None
        finally:
            # Restore original env vars
            for key, value in original_keys.items():
                os.environ[key] = value
