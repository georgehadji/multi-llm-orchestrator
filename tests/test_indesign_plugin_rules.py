#!/usr/bin/env python3
"""Test InDesign Plugin Rules Engine."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 70)
print("InDesign Plugin Rules - Import Test")
print("=" * 70)

try:
    from orchestrator import (
        InDesignPluginRules,
        InDesignRulesConfig,
        generate_indesign_plugin_rules,
    )
    print("✅ All InDesign imports successful")
except Exception as e:
    print(f"❌ Error: {e}")
