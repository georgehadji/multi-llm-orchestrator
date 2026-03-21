#!/usr/bin/env python3
"""Test Output Organizer."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 70)
print("Output Organizer - Import Test")
print("=" * 70)

try:
    from orchestrator import (
        OutputOrganizer,
        OrganizationReport,
        TestResult,
        suppress_cache_messages,
        CacheMessageSuppressor,
    )
    print("[OK] All Output Organizer imports successful")
except Exception as e:
    print(f"[ERROR] Error: {e}")
