#!/usr/bin/env python
"""Test Python syntax of all modified files."""

import py_compile
import sys

files = [
    "orchestrator/compat.py",
    "orchestrator/progress.py",
    "orchestrator/cli.py",
    "orchestrator/engine.py",
    "orchestrator/streaming.py",
    "orchestrator/nash_events.py",
    "orchestrator/nash_backup.py",
    "orchestrator/nash_auto_tuning.py",
    "orchestrator/cli_nash.py",
]

errors = []

for filepath in files:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ {filepath}")
    except Exception as e:
        print(f"✗ {filepath}: {e}")
        errors.append((filepath, e))

if errors:
    print(f"\n{len(errors)} file(s) have syntax errors:")
    for filepath, error in errors:
        print(f"  - {filepath}: {error}")
    sys.exit(1)
else:
    print("\n✅ All files have valid syntax!")
    sys.exit(0)
