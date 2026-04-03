#!/usr/bin/env python3
"""
TDD Implementation Test Script
===============================
Test the TDD implementation with a simple project.
"""

import asyncio
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_tdd_config():
    """Test TDD configuration loading."""
    print("\n" + "=" * 70)
    print("TEST 1: TDD Configuration")
    print("=" * 70)

    from orchestrator.cost_optimization import (
        TDDModelConfig,
        get_tdd_profile,
        estimate_tdd_cost,
    )

    # Test 1: Default config
    print("\n1. Testing default TDD config...")
    config = TDDModelConfig()
    assert config.test_generation == "anthropic/claude-sonnet-4-6"
    assert config.implementation == "qwen/qwen-2.5-coder-32b-instruct"
    print("   ✅ Default config loaded correctly")

    # Test 2: Get profile by tier
    print("\n2. Testing profile selection...")
    for tier in ["budget", "balanced", "premium"]:
        profile = get_tdd_profile(tier)
        assert profile is not None
        print(f"   ✅ {tier.title()} profile: {profile.test_generation}")

    # Test 3: Cost estimation
    print("\n3. Testing cost estimation...")
    for tier in ["budget", "balanced", "premium"]:
        costs = estimate_tdd_cost(tier)
        print(f"   ✅ {tier.title()}: ${costs['estimated_total_per_task']:.4f} per task")

    print("\n✅ TEST 1 PASSED: TDD Configuration working correctly")
    return True


async def test_framework_detection():
    """Test testing framework detection."""
    print("\n" + "=" * 70)
    print("TEST 2: Testing Framework Detection")
    print("=" * 70)

    from orchestrator.test_first_generator import (
        detect_testing_framework,
        get_framework_config,
        TestingFramework,
    )

    # Test cases
    test_cases = [
        # (prompt, context, extension, expected)
        ("Build a Python API with pytest", "", ".py", TestingFramework.PYTEST),
        ("Create React components with tests", "", ".js", TestingFramework.JEST),
        ("Build Vue app", "", ".ts", TestingFramework.VITEST),
        ("Go microservice", "", ".go", TestingFramework.GO_TEST),
        ("Rust library", "", ".rs", TestingFramework.CARGO_TEST),
        ("Python unittest", "", ".py", TestingFramework.UNITTEST),
    ]

    print("\nTesting framework detection...")
    for prompt, context, ext, expected in test_cases:
        result = detect_testing_framework(prompt, context, ext)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{prompt[:40]}...' + {ext} → {result.value}")
        assert result == expected, f"Expected {expected}, got {result}"

    # Test framework config
    print("\nTesting framework config...")
    for framework in [TestingFramework.PYTEST, TestingFramework.JEST, TestingFramework.VITEST]:
        config = get_framework_config(framework)
        assert "test_file_prefix" in config
        assert "test_file_suffix" in config
        assert "run_command" in config
        print(f"   ✅ {framework.value}: {config['run_command']}")

    print("\n✅ TEST 2 PASSED: Framework detection working correctly")
    return True


async def test_optimization_config():
    """Test optimization config with TDD settings."""
    print("\n" + "=" * 70)
    print("TEST 3: Optimization Config")
    print("=" * 70)

    from orchestrator.cost_optimization import (
        get_optimization_config,
        update_config,
        OptimizationConfig,
    )

    # Test 1: Get current config
    print("\n1. Testing config retrieval...")
    config = get_optimization_config()
    assert hasattr(config, "enable_tdd_first")
    assert hasattr(config, "tdd_quality_tier")
    assert hasattr(config, "tdd_max_iterations")
    assert hasattr(config, "tdd_min_test_coverage")
    print("   ✅ Config has TDD fields")

    # Test 2: Update config
    print("\n2. Testing config update...")
    config.enable_tdd_first = True
    config.tdd_quality_tier = "balanced"
    config.tdd_max_iterations = 5
    config.tdd_min_test_coverage = 0.9
    update_config(config)
    print("   ✅ Config updated successfully")

    # Test 3: Verify update
    print("\n3. Verifying config update...")
    new_config = get_optimization_config()
    assert new_config.enable_tdd_first == True
    assert new_config.tdd_quality_tier == "balanced"
    assert new_config.tdd_max_iterations == 5
    assert new_config.tdd_min_test_coverage == 0.9
    print("   ✅ Config verified")

    print("\n✅ TEST 3 PASSED: Optimization config working correctly")
    return True


async def test_tdd_generator():
    """Test TDD generator initialization."""
    print("\n" + "=" * 70)
    print("TEST 4: TDD Generator")
    print("=" * 70)

    from orchestrator.test_first_generator import TestFirstGenerator
    from orchestrator.cost_optimization import get_tdd_profile

    # Test 1: Initialize with default config
    print("\n1. Testing TDD generator initialization...")

    # Create mock client and sandbox (we won't actually call them)
    class MockClient:
        pass

    class MockSandbox:
        pass

    tdd_config = get_tdd_profile("balanced", "python")
    generator = TestFirstGenerator(
        client=MockClient(),
        sandbox=MockSandbox(),
        model_config=tdd_config,
        quality_tier="balanced",
        language="python",
    )

    assert generator.model_config is not None
    assert generator.quality_tier == "balanced"
    assert generator.language == "python"
    print("   ✅ Generator initialized with config")

    # Test 2: Model selection
    print("\n2. Testing model selection...")
    test_model = generator._get_model_for_phase("test_generation")
    impl_model = generator._get_model_for_phase("implementation")
    review_model = generator._get_model_for_phase("test_review")

    print(f"   ✅ Test generation: {test_model.value}")
    print(f"   ✅ Implementation: {impl_model.value}")
    print(f"   ✅ Test review: {review_model.value}")

    # Test 3: Cost tracking
    print("\n3. Testing cost tracking...")
    generator._track_cost("test_generation", 0.01)
    generator._track_cost("implementation", 0.02)
    generator._track_cost("review", 0.005)

    total = generator._get_total_cost()
    assert abs(total - 0.035) < 0.001
    print(f"   ✅ Cost tracking: ${total:.4f}")

    print("\n✅ TEST 4 PASSED: TDD Generator working correctly")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 TDD IMPLEMENTATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("TDD Configuration", test_tdd_config),
        ("Framework Detection", test_framework_detection),
        ("Optimization Config", test_optimization_config),
        ("TDD Generator", test_tdd_generator),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if error:
            print(f"         Error: {error}")

    print("\n" + "-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! TDD implementation is working correctly.")
        print("\nNext steps:")
        print('  1. Test with a real project: python -m orchestrator --project "..." --tdd-first')
        print("  2. Check documentation: TDD_IMPLEMENTATION_GUIDE.md")
        print("  3. Review complete summary: TDD_COMPLETE_SUMMARY.md")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
