#!/usr/bin/env python3
"""Test Front-End Rules Engine."""
import sys
sys.path.insert(0, r'E:\Documents\Vibe-Coding\Ai Orchestrator')

from orchestrator.frontend_rules import FrontendRules, FrontendConfig, generate_frontend_rules
from pathlib import Path

def main():
    print("=" * 70)
    print("Front-End Development Rules Engine - Test")
    print("=" * 70)
    
    rules = FrontendRules()
    
    # Test 1: Templates
    print("\n📚 Available Templates:")
    for key, template in rules.get_all_templates().items():
        print(f"  • {template['name']} ({key})")
    
    # Test 2: Config generation
    print("\n🎯 Testing Config Generation:")
    config = rules.generate_config("SaaS Dashboard", template="dashboard")
    print(f"  Name: {config.project_name}")
    print(f"  Slug: {config.project_slug}")
    print(f"  Stack: {config.framework}+{config.language}")
    
    print("\n" + "=" * 70)
    print("✅ Front-End Rules Engine Ready!")
    print("=" * 70)

if __name__ == "__main__":
    main()
