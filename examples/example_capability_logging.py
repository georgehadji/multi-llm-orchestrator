#!/usr/bin/env python3
"""
Example: Capability Usage Logging
=================================
Demonstrates how to use the capability logging system.
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

from orchestrator import (
    CapabilityLogger,
    CapabilityType,
    get_capability_logger,
    log_capability,
    log_capability_use,
)


def example_basic_logging():
    """Example 1: Basic capability logging."""
    print("=" * 60)
    print("Example 1: Basic Capability Logging")
    print("=" * 60)
    
    # Log a task start
    log_capability(
        capability=CapabilityType.TASK_STARTED,
        task_type="code_generation",
        model="GPT_4O",
        project_id="demo_project_001",
        details={"prompt_length": 500, "language": "python"}
    )
    
    # Simulate work
    time.sleep(0.1)
    
    # Log task completion
    log_capability(
        capability=CapabilityType.TASK_COMPLETED,
        task_type="code_generation",
        model="GPT_4O",
        project_id="demo_project_001",
        duration_ms=1500,
        success=True,
        details={"output_tokens": 250, "iterations": 2}
    )
    
    print("✓ Logged task start and completion")
    
    # Log a validation failure
    log_capability(
        capability=CapabilityType.VALIDATION_FAILED,
        task_type="code_review",
        model="DEEPSEEK_CODER",
        project_id="demo_project_001",
        duration_ms=500,
        success=False,
        details={"validator": "python_syntax", "error": "IndentationError"}
    )
    
    print("✓ Logged validation failure")


@log_capability_use(CapabilityType.CODEBASE_ANALYSIS, task_type_arg="analysis_type")
def analyze_codebase(path: str, analysis_type: str = "general"):
    """Example function with automatic capability logging."""
    print(f"  Analyzing {path} ({analysis_type})...")
    time.sleep(0.05)
    return {"files": 42, "languages": ["python", "javascript"]}


def example_decorator():
    """Example 2: Using the decorator for automatic logging."""
    print("\n" + "=" * 60)
    print("Example 2: Automatic Logging with Decorator")
    print("=" * 60)
    
    result = analyze_codebase("./my_project", analysis_type="deep")
    print(f"✓ Analysis complete: {result}")


def example_statistics():
    """Example 3: Retrieving capability statistics."""
    print("\n" + "=" * 60)
    print("Example 3: Capability Statistics")
    print("=" * 60)
    
    # Flush any pending logs
    logger = get_capability_logger()
    logger.flush()
    
    # Get stats
    stats = logger.get_stats()
    
    print(f"Total events: {stats['total_events']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average duration: {stats['avg_duration_ms']:.1f}ms")
    print(f"\nBy capability:")
    for cap, count in stats['by_capability'].items():
        print(f"  - {cap}: {count}")


def example_direct_logger():
    """Example 4: Using the logger directly for batch operations."""
    print("\n" + "=" * 60)
    print("Example 4: Direct Logger Usage")
    print("=" * 60)
    
    logger = CapabilityLogger()
    
    # Log multiple cache operations
    for i in range(5):
        logger.log(
            capability=CapabilityType.CACHE_HIT,
            details={"key": f"cache_key_{i}", "hit_count": i + 1}
        )
    
    logger.log(
        capability=CapabilityType.CACHE_MISS,
        details={"key": "missing_key", "fallback": "database"}
    )
    
    print("✓ Logged 5 cache hits and 1 cache miss")
    
    # Flush to disk
    logger.flush()
    print("✓ Flushed to disk")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " Capability Usage Logging Demo ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        example_basic_logging()
        example_decorator()
        example_direct_logger()
        example_statistics()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print(f"\nLog files location: ./logs/capabilities/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
