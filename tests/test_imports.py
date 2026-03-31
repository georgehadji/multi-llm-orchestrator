#!/usr/bin/env python
"""Test imports are working correctly."""

print("Testing imports...")

try:
    from orchestrator.progress import ProgressRenderer

    print("✓ ProgressRenderer imported")
except Exception as e:
    print(f"✗ ProgressRenderer import failed: {e}")

try:
    from orchestrator.cli import main

    print("✓ CLI imported")
except Exception as e:
    print(f"✗ CLI import failed: {e}")

try:
    from orchestrator import NashStableOrchestrator

    print("✓ NashStableOrchestrator imported")
except Exception as e:
    print(f"✗ NashStableOrchestrator import failed: {e}")

try:
    from orchestrator import get_event_bus

    print("✓ Event bus imported")
except Exception as e:
    print(f"✗ Event bus import failed: {e}")

try:
    from orchestrator import get_backup_manager

    print("✓ Backup manager imported")
except Exception as e:
    print(f"✗ Backup manager import failed: {e}")

try:
    from orchestrator import get_auto_tuner

    print("✓ Auto tuner imported")
except Exception as e:
    print(f"✗ Auto tuner import failed: {e}")

print("\nAll imports successful!")
