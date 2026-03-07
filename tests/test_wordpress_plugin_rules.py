#!/usr/bin/env python3
"""Test WordPress Plugin Rules Engine."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

print("=" * 70)
print("WordPress Plugin Rules - Import Test")
print("=" * 70)

try:
    from orchestrator import (
        WordPressPluginRules,
        WPRulesConfig,
        generate_wordpress_plugin_rules,
    )
    print("✅ All WordPress imports successful")
except Exception as e:
    print(f"❌ Error: {e}")
