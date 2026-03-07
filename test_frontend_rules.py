#!/usr/bin/env python3
"""Test Front-End Rules Engine."""
import sys
sys.path.insert(0, 'E:\\Documents\\Vibe-Coding\\Ai Orchestrator')

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
        print(f"\n  {template['name']} ({key})")
        print(f"    {template['description']}")
        print(f"    Features: {', '.join(template['features'][:3])}...")
    
    # Test 2: Generate config for each template
    print("\n\n🎯 Testing Config Generation:")
    templates = ['saas', 'dashboard', 'ai_first', 'minimal', 'microfrontend']
    
    for template in templates:
        config = rules.generate_config(f"Test {template.title()}", template=template)
        print(f"\n  {template.upper()}:")
        print(f"    Name: {config.project_name}")
        print(f"    Slug: {config.project_slug}")
        print(f"    Stack: {config.framework}+{config.language}")
        print(f"    State: {config.server_state}+{config.client_state}")
        print(f"    Tests: {config.test_framework}+{config.e2e_framework}")
    
    # Test 3: Generate rules file content
    print("\n\n📝 Testing Rules File Generation:")
    config = rules.generate_config("SaaS Dashboard", template="dashboard")
    content = rules.get_rules_file_content(config)
    
    # Preview first 500 chars
    preview = content[:500].replace('\n', ' ')
    print(f"  Preview: {preview}...")
    print(f"  Total length: {len(content)} characters")
    
    # Test 4: Save to file
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    rules_file = rules.save_rules_file(config, output_dir)
    print(f"\n  ✅ Rules file saved to: {rules_file}")
    
    # Test 5: Convenience function
    print("\n\n🔧 Testing Convenience Function:")
    rules_file2 = generate_frontend_rules(
        "AI Chat Application",
        output_dir,
        template="ai_first",
        styling="antd",
    )
    print(f"  ✅ Generated with convenience function: {rules_file2}")
    
    print("\n\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
