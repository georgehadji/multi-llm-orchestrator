"""
Adversarial Validation Tests for Bug Fixes
==========================================
Tests the fixes for BUG-CONC-001, BUG-ROUTER-003, and BUG-BUDGET-004
"""
import asyncio
import time
from orchestrator.models import Budget
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model


async def test_budget_concurrent_charge():
    """
    ATTACK VECTOR: Multiple concurrent tasks charge budget simultaneously.
    EXPECTED: All charges applied correctly, no race condition.
    """
    print("\n[TEST 1] Budget Concurrent Charge Attack")
    budget = Budget(max_usd=100.0)
    
    # Verify lock is eagerly initialized
    assert isinstance(budget._lock, asyncio.Lock), "Lock must be asyncio.Lock"
    print("  ✓ Lock eagerly initialized")
    
    # Simulate 10 concurrent tasks each charging $5
    async def charge_task(task_id: int):
        await budget.charge(5.0, "generation")
        return task_id
    
    tasks = [charge_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    expected_total = 50.0
    actual_total = budget.spent_usd
    
    print(f"  Expected total: ${expected_total:.2f}")
    print(f"  Actual total: ${actual_total:.2f}")
    
    assert abs(actual_total - expected_total) < 0.001, f"Budget race condition! Expected {expected_total}, got {actual_total}"
    print("  ✓ PASS: All charges applied correctly\n")
    return True


async def test_budget_reservation_failure_recovery():
    """
    ATTACK VECTOR: commit_reservation() called, charge() fails.
    EXPECTED: Reservation restored, no budget leakage.
    """
    print("[TEST 2] Budget Reservation Failure Recovery Attack")
    budget = Budget(max_usd=100.0)
    
    # Reserve $10
    reserved = await budget.reserve(10.0)
    assert reserved, "Reservation should succeed"
    assert budget._reserved_usd == 10.0, f"Expected $10 reserved, got ${budget._reserved_usd}"
    print(f"  ✓ Reserved $10.00, _reserved_usd = ${budget._reserved_usd}")
    
    # Mock charge to fail
    original_charge = budget.charge
    charge_failed = False
    
    async def failing_charge(amount, phase="generation"):
        nonlocal charge_failed
        charge_failed = True
        raise Exception("Simulated charge failure")
    
    budget.charge = failing_charge
    
    # Try to commit - should fail and restore reservation
    try:
        await budget.commit_reservation(8.0)
        assert False, "commit_reservation should have raised exception"
    except Exception as e:
        print(f"  ✓ commit_reservation raised exception as expected: {e}")
    
    # Verify reservation was restored
    print(f"  After failure: _reserved_usd = ${budget._reserved_usd}")
    assert budget._reserved_usd == 10.0, f"Reservation should be restored to $10, got ${budget._reserved_usd}"
    print("  ✓ PASS: Reservation restored after charge failure\n")
    
    # Restore original charge
    budget.charge = original_charge
    return True


async def test_router_concurrent_read_write():
    """
    ATTACK VECTOR: Concurrent reads (is_available) while writing (record_timeout).
    EXPECTED: No data races, consistent state.
    """
    print("[TEST 3] Router Concurrent Read/Write Attack")
    router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=60.0)
    
    model = Model.GPT_4O
    errors = []
    
    # Writer: continuously record timeouts
    async def writer():
        for _ in range(100):
            await router.record_timeout(model)
            await asyncio.sleep(0.001)
    
    # Reader: continuously check availability
    async def reader():
        for _ in range(100):
            try:
                # Use new async-safe method
                available = await router.is_available_async(model)
                # Also test sync method
                available_sync = router.is_available(model)
            except Exception as e:
                errors.append(f"Reader error: {e}")
            await asyncio.sleep(0.001)
    
    # Run concurrently
    await asyncio.gather(writer(), reader())
    
    if errors:
        print(f"  ✗ FAIL: Errors during concurrent access: {errors}")
        return False
    
    # Verify final state is consistent
    state = await router.get_state(model)
    print(f"  Final model state: {state.value}")
    assert state == ModelState.DEGRADED, f"Model should be DEGRADED after 100 timeouts, got {state.value}"
    print("  ✓ PASS: No data races, state consistent\n")
    return True


async def test_budget_extreme_concurrency():
    """
    ATTACK VECTOR: 100 concurrent tasks all trying to reserve/charge simultaneously.
    EXPECTED: No deadlocks, correct final state.
    """
    print("[TEST 4] Budget Extreme Concurrency Attack (100 tasks)")
    budget = Budget(max_usd=1000.0)
    
    async def worker(task_id: int):
        # Reserve, do work, commit
        if await budget.reserve(5.0):
            await asyncio.sleep(0.001)  # Simulate work
            await budget.commit_reservation(4.5, "generation")
            return task_id
        return None
    
    tasks = [worker(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    
    successful = [r for r in results if r is not None]
    print(f"  Successful tasks: {len(successful)}")
    print(f"  Final spent: ${budget.spent_usd:.2f}")
    print(f"  Final reserved: ${budget._reserved_usd:.2f}")
    
    expected_spent = len(successful) * 4.5
    assert abs(budget.spent_usd - expected_spent) < 0.01, f"Expected ${expected_spent:.2f}, got ${budget.spent_usd:.2f}"
    assert budget._reserved_usd == 0.0, f"Expected $0 reserved, got ${budget._reserved_usd}"
    print("  ✓ PASS: Extreme concurrency handled correctly\n")
    return True


async def test_router_auth_failure_isolation():
    """
    ATTACK VECTOR: Auth failure on one model shouldn't affect others.
    EXPECTED: Only the failed model is disabled.
    """
    print("[TEST 5] Router Auth Failure Isolation Attack")
    router = AdaptiveRouter()
    
    # Disable one model
    await router.record_auth_failure(Model.GPT_4O)
    state_gpt4o = await router.get_state(Model.GPT_4O)
    state_gpt4o_mini = await router.get_state(Model.GPT_4O_MINI)
    
    print(f"  GPT-4O state: {state_gpt4o.value}")
    print(f"  GPT-4O-Mini state: {state_gpt4o_mini.value}")
    
    assert state_gpt4o == ModelState.DISABLED, "GPT-4O should be DISABLED"
    assert state_gpt4o_mini == ModelState.HEALTHY, "GPT-4O-Mini should be HEALTHY"
    
    # Test is_available_async
    avail_gpt4o = await router.is_available_async(Model.GPT_4O)
    avail_gpt4o_mini = await router.is_available_async(Model.GPT_4O_MINI)
    
    print(f"  GPT-4O available: {avail_gpt4o}")
    print(f"  GPT-4O-Mini available: {avail_gpt4o_mini}")
    
    assert avail_gpt4o == False, "GPT-4O should not be available"
    assert avail_gpt4o_mini == True, "GPT-4O-Mini should be available"
    print("  ✓ PASS: Auth failure properly isolated\n")
    return True


async def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION - Bug Fix Stress Tests")
    print("=" * 60)
    
    tests = [
        ("Concurrent Budget Charge", test_budget_concurrent_charge),
        ("Reservation Failure Recovery", test_budget_reservation_failure_recovery),
        ("Router Read/Write Race", test_router_concurrent_read_write),
        ("Extreme Budget Concurrency", test_budget_extreme_concurrency),
        ("Router Auth Isolation", test_router_auth_failure_isolation),
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
