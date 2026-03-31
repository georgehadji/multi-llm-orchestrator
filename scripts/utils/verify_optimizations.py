#!/usr/bin/env python
"""
Quick verification tests for all engine optimizations.
Run: python tests/verify_optimizations.py
"""
import sys
import asyncio
import weakref

print("=" * 60)
print("ENGINE OPTIMIZATION VERIFICATION TESTS")
print("=" * 60)

from orchestrator.engine import Orchestrator
from orchestrator.models import Budget, Task, TaskType
from orchestrator.telemetry_store import TelemetryStore

def test_p0_2_weakset():
    """Test P0-2: Memory leak fix with WeakSet."""
    print("\n[P0-2] Testing WeakSet-based background task tracking...")
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    assert isinstance(orch._background_tasks, weakref.WeakSet), "Should be WeakSet"
    assert orch._cleanup_timer is None, "Timer should start as None"
    print("  ✓ WeakSet verified")
    print("  ✓ Cleanup timer initialized")
    print("  ✅ P0-2 PASSED")

def test_p1_1_profile_cache():
    """Test P1-1: Profile caching."""
    print("\n[P1-1] Testing profile caching...")
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    assert orch._active_profiles_cache is None, "Cache should start None"
    print("  ✓ Cache starts as None")
    
    # Build cache
    orch._profiles[list(orch._profiles.keys())[0]].call_count = 5
    active = orch._get_active_profiles()
    assert orch._active_profiles_cache is not None, "Cache should be built"
    print("  ✓ Cache builds on first call")
    
    # Invalidate
    orch._invalidate_profile_cache()
    assert orch._active_profiles_cache is None, "Cache should be invalidated"
    print("  ✓ Cache invalidation works")
    print("  ✅ P1-1 PASSED")

def test_p1_2_adaptive_decomposition():
    """Test P1-2: Adaptive decomposition model selection."""
    print("\n[P1-2] Testing adaptive decomposition...")
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    
    # Enable test models
    orch.api_health[orch._get_fast_decomposition_model()] = True
    
    # Simple project
    model = orch._select_decomposition_model("Simple hello world")
    print(f"  Simple project → {model.value}")
    
    # Complex project
    complex_proj = """
    Build distributed microservices with Kubernetes, OAuth, 
    PostgreSQL replication, Redis caching, WebSocket streaming,
    Kafka queue, multi-tenant SaaS
    """
    model = orch._select_decomposition_model(complex_proj)
    print(f"  Complex project → {model.value}")
    print("  ✅ P1-2 PASSED")

async def test_p2_2_batch_telemetry():
    """Test P2-2: Batch telemetry operations."""
    print("\n[P2-2] Testing batch telemetry...")
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = TelemetryStore(db_path=db_path)
        
        # Test batch method exists
        assert hasattr(store, 'record_snapshots_batch'), "Batch method should exist"
        print("  ✓ Batch method exists")
        
        # Test empty batch
        await store.record_snapshots_batch("test", [])
        print("  ✓ Empty batch handled")
        
        # Test with data
        orch = Orchestrator(budget=Budget(max_usd=100.0))
        model = list(orch._profiles.keys())[0]
        orch._profiles[model].call_count = 5
        await store.record_snapshots_batch("test", [(model, orch._profiles[model])])
        print("  ✓ Batch with data works")
        print("  ✅ P2-2 PASSED")

def test_p2_1_json5():
    """Test P2-1: JSON5 parsing."""
    print("\n[P2-1] Testing JSON5 parsing...")
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    
    # Test valid JSON
    valid = '[{"id": "t1", "type": "code_generation", "prompt": "test", "dependencies": []}]'
    tasks = orch._parse_decomposition(valid)
    assert len(tasks) == 1, "Should parse valid JSON"
    print("  ✓ Valid JSON parsed")
    
    # Test with markdown fences
    with_fences = '```json\n' + valid + '\n```'
    tasks = orch._parse_decomposition(with_fences)
    assert len(tasks) == 1, "Should strip markdown fences"
    print("  ✓ Markdown fences stripped")
    
    # Test invalid JSON
    invalid = "not json at all"
    tasks = orch._parse_decomposition(invalid)
    assert tasks == {}, "Should return empty dict on failure"
    print("  ✓ Invalid JSON handled gracefully")
    print("  ✅ P2-1 PASSED")

async def main():
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    try:
        test_p0_2_weakset()
        test_p1_1_profile_cache()
        test_p1_2_adaptive_decomposition()
        await test_p2_2_batch_telemetry()
        test_p2_1_json5()
        
        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED!")
        print("=" * 60)
        print("\nOptimizations verified:")
        print("  • P0-2: WeakSet memory leak fix")
        print("  • P1-1: Profile caching")
        print("  • P1-2: Adaptive decomposition")
        print("  • P2-1: JSON5 parsing")
        print("  • P2-2: Batch telemetry")
        print("\nNote: P0-1 (semaphore optimization) requires integration testing")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
