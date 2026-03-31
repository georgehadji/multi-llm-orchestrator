#!/usr/bin/env python
"""Test basic imports are working."""

import sys


def test_import(module_name, item_name=None):
    try:
        if item_name:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"✓ {module_name}.{item_name}")
        else:
            __import__(module_name)
            print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}{('.' + item_name) if item_name else ''}: {e}")
        return False


print("Testing imports...\n")

# Test core modules
test_import("orchestrator")
test_import("orchestrator.models")
test_import("orchestrator.cli")
test_import("orchestrator.progress")
test_import("orchestrator.engine")
test_import("orchestrator.streaming")

# Test Nash stability modules
test_import("orchestrator.knowledge_graph")
test_import("orchestrator.adaptive_templates")
test_import("orchestrator.pareto_frontier")
test_import("orchestrator.federated_learning")
test_import("orchestrator.nash_stable_orchestrator")
test_import("orchestrator.nash_events")
test_import("orchestrator.nash_backup")
test_import("orchestrator.nash_auto_tuning")

# Test CLI
from orchestrator.cli import main

print("\n✓ CLI main function imported successfully")

print("\n✅ All imports successful!")
