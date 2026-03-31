"""
Adversarial Validation Tests for Bug Fixes
==========================================
Tests the fixes for BUG-API-001, BUG-CACHE-002, and BUG-STATE-003
"""

import asyncio
from orchestrator.api_clients import UnifiedClient
from orchestrator.cache import DiskCache
from orchestrator.state import StateManager


async def test_api_client_init_no_nameerror():
    """
    ATTACK VECTOR: Instantiate UnifiedClient without os module available.
    EXPECTED: No NameError, client initializes successfully.
    """
    print("\n[TEST 1] UnifiedClient Initialization Attack")

    # This would have raised NameError before the fix
    client = UnifiedClient()

    assert client is not None
    assert hasattr(client, "xai_region")
    print("  ✓ PASS: No NameError, client initialized\n")
    return True


async def test_cache_concurrent_access():
    """
    ATTACK VECTOR: Multiple concurrent cache operations before lock init.
    EXPECTED: All operations serialized correctly, no race condition.
    """
    print("[TEST 2] Cache Concurrent Access Attack")

    cache = DiskCache()

    # Verify lock is eagerly initialized
    assert isinstance(cache._lock, asyncio.Lock), "Lock must be asyncio.Lock"
    print("  ✓ Lock eagerly initialized")

    # Simulate 20 concurrent get/put operations
    async def worker(i: int):
        await cache.put(f"model_{i}", f"prompt_{i}", 100, f"response_{i}")
        result = await cache.get(f"model_{i}", f"prompt_{i}", 100)
        return result is not None

    tasks = [worker(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r)
    print(f"  Successful operations: {success_count}/20")

    assert success_count == 20, f"Expected 20 successes, got {success_count}"
    print("  ✓ PASS: All concurrent operations succeeded\n")
    return True


async def test_state_concurrent_access():
    """
    ATTACK VECTOR: Multiple concurrent state saves before lock init.
    EXPECTED: All operations serialized correctly, no corruption.
    """
    print("[TEST 3] State Manager Concurrent Access Attack")

    state_mgr = StateManager()

    # Verify lock is eagerly initialized
    assert isinstance(state_mgr._lock, asyncio.Lock), "Lock must be asyncio.Lock"
    print("  ✓ Lock eagerly initialized")

    # Simulate 10 concurrent connection requests
    connections = []
    errors = []

    async def get_connection(i: int):
        try:
            conn = await state_mgr._get_conn()
            connections.append((i, conn))
        except Exception as e:
            errors.append((i, e))

    tasks = [get_connection(i) for i in range(10)]
    await asyncio.gather(*tasks)

    print(f"  Connection requests: {len(connections)}")
    print(f"  Errors: {len(errors)}")

    # All should get the same connection (singleton pattern)
    if connections:
        first_conn = connections[0][1]
        same_conn_count = sum(1 for _, c in connections if c is first_conn)
        print(f"  Same connection reused: {same_conn_count}/{len(connections)}")
        assert same_conn_count == len(connections), "All should share same connection"

    assert len(errors) == 0, f"Errors occurred: {errors}"
    print("  ✓ PASS: All concurrent access succeeded\n")
    return True


async def test_cache_extreme_concurrency():
    """
    ATTACK VECTOR: 100 concurrent cache operations simultaneously.
    EXPECTED: No deadlocks, correct final state.
    """
    print("[TEST 4] Cache Extreme Concurrency Attack (100 ops)")

    cache = DiskCache()

    async def worker(task_id: int):
        key = f"key_{task_id}"
        await cache.put("test-model", key, 100, f"value_{task_id}")
        result = await cache.get("test-model", key, 100)
        return result is not None

    tasks = [worker(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r)
    print(f"  Successful operations: {success_count}/100")

    assert success_count == 100, f"Expected 100 successes, got {success_count}"
    print("  ✓ PASS: Extreme concurrency handled correctly\n")
    return True


async def test_api_client_xai_region_env():
    """
    ATTACK VECTOR: XAI_REGION environment variable handling.
    EXPECTED: Correctly reads from os.environ.
    """
    print("[TEST 5] XAI Region Environment Variable Attack")

    import os

    # Test with no XAI_REGION set
    if "XAI_REGION" in os.environ:
        del os.environ["XAI_REGION"]

    client1 = UnifiedClient()
    assert client1.xai_region is None, f"Expected None, got {client1.xai_region}"
    print("  ✓ No XAI_REGION: correctly returns None")

    # Test with XAI_REGION set
    os.environ["XAI_REGION"] = "us-east-1"
    client2 = UnifiedClient()
    assert client2.xai_region == "us-east-1", f"Expected us-east-1, got {client2.xai_region}"
    print("  ✓ XAI_REGION=us-east-1: correctly read from env")

    # Cleanup
    del os.environ["XAI_REGION"]
    print("  ✓ PASS: Environment variable handling correct\n")
    return True


async def test_cache_lock_single_instance():
    """
    ATTACK VECTOR: Verify lock is truly single instance, not recreated.
    EXPECTED: Same lock object used throughout lifetime.
    """
    print("[TEST 6] Cache Lock Identity Attack")

    cache = DiskCache()
    lock_id_1 = id(cache._lock)

    # Access connection multiple times
    await cache._get_conn()
    lock_id_2 = id(cache._lock)

    await cache.get("test", "test", 100)
    lock_id_3 = id(cache._lock)

    assert lock_id_1 == lock_id_2 == lock_id_3, "Lock identity must remain constant"
    print("  ✓ Lock identity constant across operations")
    print("  ✓ PASS: Lock is truly singleton\n")
    return True


async def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION - Bug Fix Stress Tests")
    print("=" * 60)

    tests = [
        ("UnifiedClient Init", test_api_client_init_no_nameerror),
        ("Cache Concurrent Access", test_cache_concurrent_access),
        ("State Concurrent Access", test_state_concurrent_access),
        ("Cache Extreme Concurrency", test_cache_extreme_concurrency),
        ("XAI Region Env Var", test_api_client_xai_region_env),
        ("Cache Lock Identity", test_cache_lock_single_instance),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAIL: {e}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r, _ in results if r)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n✓ ALL ATTACKS REPELLED - Fixes are robust")
    else:
        print(f"\n✗ {total - passed} attacks succeeded - Fixes need strengthening")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
