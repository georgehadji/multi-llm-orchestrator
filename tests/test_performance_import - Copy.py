#!/usr/bin/env python3
"""Quick test to verify all performance modules can be imported."""
import sys
import traceback

def test_import(module_name, items=None):
    """Test importing a module."""
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"✓ {module_name}")
        
        if items:
            for item in items:
                if hasattr(module, item):
                    print(f"  ✓ {item}")
                else:
                    print(f"  ✗ {item} (not found)")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        traceback.print_exc()
        return False

print("Testing Performance Module Imports")
print("=" * 50)

results = []

# Test performance module
results.append(test_import("orchestrator.performance", [
    "LRUCache",
    "RedisCache",
    "ConnectionPool",
    "MetricsCollector",
    "QueryOptimizer",
    "cached",
    "cache_invalidate",
    "get_cache",
]))

# Test monitoring module
results.append(test_import("orchestrator.monitoring", [
    "MetricsRegistry",
    "KPIReporter",
    "KPITier",
    "HealthChecker",
    "monitor_endpoint",
    "monitor_async_task",
    "STANDARD_KPIS",
    "health_checker",
    "metrics",
]))

# Test optimized dashboard
results.append(test_import("orchestrator.dashboard_optimized", [
    "OptimizedDashboardServer",
    "PerformanceConfig",
    "CacheManager",
    "DebouncedUpdater",
    "PerformanceMonitor",
    "EXTERNAL_CSS",
]))

print("=" * 50)

if all(results):
    print("✓ All modules imported successfully!")
    print("\nNext steps:")
    print("  1. Run: python run_optimized_dashboard.py")
    print("  2. Test: pytest tests/test_performance.py -v")
    sys.exit(0)
else:
    print("✗ Some imports failed")
    sys.exit(1)
