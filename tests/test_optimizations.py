"""
Regression validation for v6.0 optimizations.
Tests that all 4 optimizations are properly integrated.
"""

import sys
sys.path.insert(0, r'D:\Vibe-Coding\Ai Orchestrator')

def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")
    from orchestrator.engine import Orchestrator
    from orchestrator.semantic_cache import SemanticCache, DuplicationDetector
    from orchestrator.telemetry import _EMA_ALPHA
    from orchestrator.validators import VALIDATORS
    print("✓ All imports successful (DuplicationDetector is backward compatible)")
    return True

def test_ema_alpha():
    """Test that EMA alpha was increased for faster detection."""
    print("Testing EMA alpha...")
    from orchestrator.telemetry import _EMA_ALPHA
    assert _EMA_ALPHA == 0.2, f"Expected 0.2, got {_EMA_ALPHA}"
    print(f"✓ EMA_ALPHA = {_EMA_ALPHA} (optimized for faster detection)")
    return True

def test_tool_safety_validator():
    """Test that tool safety validator was added."""
    print("Testing tool safety validator...")
    from orchestrator.validators import VALIDATORS, validate_tool_safety
    assert "tool_safety" in VALIDATORS, "tool_safety not in VALIDATORS"
    
    # Test detection of suspicious patterns
    bad_code = "import os\nos.system('rm -rf /')"
    result = validate_tool_safety(bad_code)
    assert not result.passed, "Should detect dangerous code"
    
    # Test safe code passes
    safe_code = "print('hello world')"
    result = validate_tool_safety(safe_code)
    assert result.passed, "Safe code should pass"
    
    print("✓ Tool safety validator working")
    return True

def test_semantic_cache():
    """Test semantic cache functionality."""
    print("Testing semantic cache...")
    from orchestrator.semantic_cache import SemanticCache
    from orchestrator.models import Task, TaskType
    
    cache = SemanticCache(quality_threshold=0.85)
    stats = cache.get_stats()
    assert stats["entries"] == 0, "New cache should be empty"
    
    # Create a test task
    task = Task(
        id="test_001",
        type=TaskType.CODE_GEN,
        prompt="Generate a function to add two numbers",
    )
    
    # Cache a pattern
    cached = cache.cache_pattern(task, "def add(a, b): return a + b", 0.90)
    assert cached, "Should cache high-quality result"
    
    # Check stats updated
    stats = cache.get_stats()
    assert stats["entries"] == 1, "Cache should have 1 entry"
    
    print("✓ Semantic cache working")
    return True

def test_early_exit_logic():
    """Test confidence-based early exit logic."""
    print("Testing early exit logic...")
    from orchestrator.engine import Orchestrator
    
    # Create minimal instance for method testing
    orch = Orchestrator.__new__(Orchestrator)
    
    # Test: not enough history
    result = orch._should_exit_early([0.90], 0.85)
    assert not result, "Should not exit with only 1 score"
    
    # Test: stable high performance
    result = orch._should_exit_early([0.88, 0.89, 0.90], 0.85)
    assert result, "Should exit with stable high scores"
    
    # Test: low scores should not exit
    result = orch._should_exit_early([0.50, 0.55], 0.85)
    assert not result, "Should not exit with low scores"
    
    # Test: high variance should not exit
    result = orch._should_exit_early([0.60, 0.95], 0.85)
    assert not result, "Should not exit with high variance"
    
    print("✓ Early exit logic working correctly")
    return True

def test_tiered_selection():
    """Test tiered model selection is configured."""
    print("Testing tiered selection...")
    from orchestrator.engine import Orchestrator
    from orchestrator.models import Model
    
    # Verify tiers are defined
    assert len(Orchestrator._TIER_CHEAP) > 0, "Cheap tier should not be empty"
    assert len(Orchestrator._TIER_BALANCED) > 0, "Balanced tier should not be empty"
    assert len(Orchestrator._TIER_PREMIUM) > 0, "Premium tier should not be empty"
    
    # Verify Gemini Flash Lite is in cheap tier
    assert Model.GEMINI_FLASH_LITE in Orchestrator._TIER_CHEAP, "Flash Lite should be cheap"
    
    print("✓ Tiered selection configured")
    return True

def main():
    """Run all regression tests."""
    print("=" * 60)
    print("OPTIMIZATION REGRESSION VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ema_alpha,
        test_tool_safety_validator,
        test_semantic_cache,
        test_early_exit_logic,
        test_tiered_selection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
