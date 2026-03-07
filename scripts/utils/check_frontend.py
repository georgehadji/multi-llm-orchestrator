#!/usr/bin/env python3
"""Check Front-End Rules Engine functionality."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

# Import test
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
    print(f"    └─ {template['description'][:50]}...")

# Test config generation
print("\n🎯 Config Generation:")
config = rules.generate_config("SaaS Dashboard", template="dashboard")
print(f"  Name: {config.project_name}")
print(f"  Slug: {config.project_slug}")
print(f"  Template: {config.template}")
print(f"  Stack: {config.framework} + {config.language} + {config.bundler}")
print(f"  State: {config.server_state} + {config.client_state}")
print(f"  Testing: {config.test_framework} + {config.e2e_framework}")

# Generate and save rules
print("\n📝 Generating Rules File:")
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

rules_file = rules.save_rules_file(config, output_dir)
print(f"  ✅ Saved: {rules_file}")
print(f"  Size: {rules_file.stat().st_size:,} bytes")

# Preview
content = rules_file.read_text()[:800]
print(f"\n  Preview (first 800 chars):")
print("  " + "-" * 60)
for line in content.split('\n')[:15]:
    print(f"  {line}")
print("  " + "-" * 60)

print("\n" + "=" * 70)
print("✅ Front-End Rules Engine Ready for Use!")
print("=" * 70)
