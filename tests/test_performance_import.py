#!/usr/bin/env python3
"""Test performance module imports."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

try:
    from orchestrator import PerformanceMonitor
    print("✅ PerformanceMonitor imported")
except ImportError as e:
    print(f"⚠️ Performance module: {e}")
