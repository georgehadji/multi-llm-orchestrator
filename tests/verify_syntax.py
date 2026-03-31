#!/usr/bin/env python3
"""Verify Python syntax without executing."""

import py_compile
import sys

files_to_check = [
    r"E:\Documents\Vibe-Coding\Ai Orchestrator\orchestrator\frontend_rules.py",
]

all_ok = True

for filepath in files_to_check:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✅ {filepath.split('\\')[-1]} - Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"❌ {filepath.split('\\')[-1]} - Syntax Error: {e}")
        all_ok = False

if all_ok:
    print("\n✅ All files have valid Python syntax!")
    sys.exit(0)
else:
    sys.exit(1)
