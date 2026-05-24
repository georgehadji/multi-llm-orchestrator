"""
Test Instructor + Tenacity Integration
========================================
Quick test to verify structured outputs and retry logic work correctly.

Usage:
    python -m orchestrator.test_instructor_tenacity
"""

import asyncio
import os
from orchestrator.structured_outputs import (
    decompose_project,
)
from orchestrator.retry_utils import async_retry, llm_retry, TimeoutError, RateLimitError


async def test_instructor_decomposition():
    """Test Instructor structured decomposition"""
    print("\n=== Testing Instructor Decomposition ===")

    project_desc = "Build a simple todo app with React and FastAPI"
    criteria = "Working todo app with CRUD operations"

    try:
        # Use FREE model
        tasks, order = await decompose_project(
            project_desc, criteria, model="nvidia/nemotron-3-super-120b-a12b:free"
        )

        print(f"✅ Decomposition succeeded!")
        print(f"   Tasks: {len(tasks)}")
        print(f"   Order: {order[:3]}...")  # Show first 3
        return True

    except Exception as e:
        print(f"❌ Decomposition failed: {e}")
        return False


async def test_tenacity_retry():
    """Test Tenacity retry logic"""
    print("\n=== Testing Tenacity Retry Logic ===")

    attempt_count = 0

    @async_retry(
        max_attempts=3,
        min_wait=0.1,
        max_wait=1.0,
        retryable_exceptions=(TimeoutError, RateLimitError),
    )
    async def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1

        if attempt_count < 3:
            raise TimeoutError(f"Simulated timeout (attempt {attempt_count})")

        return "Success!"

    try:
        result = await flaky_operation()
        print(f"✅ Retry succeeded after {attempt_count} attempts!")
        print(f"   Result: {result}")
        return True

    except Exception as e:
        print(f"❌ Retry failed: {e}")
        return False


async def test_llm_retry():
    """Test LLM-specific retry decorator"""
    print("\n=== Testing LLM Retry ===")

    attempt_count = 0

    @llm_retry
    async def simulated_llm_call():
        nonlocal attempt_count
        attempt_count += 1

        # Simulate rate limit on first attempt
        if attempt_count == 1:
            raise RateLimitError("Rate limit exceeded")

        return "LLM response"

    try:
        result = await simulated_llm_call()
        print(f"✅ LLM retry succeeded after {attempt_count} attempts!")
        print(f"   Result: {result}")
        return True

    except Exception as e:
        print(f"❌ LLM retry failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Instructor + Tenacity Integration Tests")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️  OPENROUTER_API_KEY not set - skipping LLM tests")
        print("   Set with: export OPENROUTER_API_KEY='your-key'")
        return

    # Run tests
    results = []

    # Test 1: Tenacity retry (no API needed)
    results.append(("Tenacity Retry", await test_tenacity_retry()))

    # Test 2: LLM retry (no API needed)
    results.append(("LLM Retry", await test_llm_retry()))

    # Test 3: Instructor decomposition (API needed)
    results.append(("Instructor Decomposition", await test_instructor_decomposition()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Instructor + Tenacity integration working!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
