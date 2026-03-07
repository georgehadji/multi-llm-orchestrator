#!/usr/bin/env python3
"""Check Front-End Rules Engine functionality."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

from orchestrator.frontend_rules import FrontendRules, FrontendConfig, generate_frontend_rules
from pathlib import Path

print("=" * 70)
print("✅ Front-End Rules Engine - Import Test")
print("=" * 70)

rules = FrontendRules()

# Templates
print("\n📚 Available Templates:")
for key, template in rules.get_all_templates().items():
    print(f"  • {template['name']} ({key})")

# Test config generation
print("\n🎯 Config Generation:")
config = rules.generate_config("SaaS Dashboard", template="dashboard")
print(f"  Name: {config.project_name}")
print(f"  Slug: {config.project_slug}")
print(f"  Stack: {config.framework} + {config.language}")

print("\n" + "=" * 70)
print("✅ Front-End Rules Engine Ready!")
print("=" * 70)
