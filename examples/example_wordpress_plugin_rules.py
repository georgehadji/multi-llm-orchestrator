"""
Example: WordPress Plugin Rules
===============================
Demonstrates how to generate WordPress plugin development rules.

Usage:
    python example_wordpress_plugin_rules.py --demo
    python example_wordpress_plugin_rules.py --generate "My Plugin"
"""
import argparse
import sys
from pathlib import Path


def demo_architecture_paths():
    """Show all architecture paths."""
    from orchestrator.wordpress_plugin_rules import WordPressPluginRules
    
    rules = WordPressPluginRules()
    
    print("=" * 70)
    print("WordPress Plugin Architecture Paths")
    print("=" * 70)
    
    for key, path in rules.get_all_architecture_paths().items():
        print(f"\n🏗️  {path['name']}")
        print(f"   Key: {key}")
        print(f"   {path['description']}")
        print(f"\n   ✅ Pros:")
        for pro in path['pros']:
            print(f"      • {pro}")
        print(f"\n   ⚠️  Cons:")
        for con in path['cons']:
            print(f"      • {con}")
        print(f"\n   📁 Structure:")
        for item in path['structure'][:5]:
            print(f"      • {item}")


def demo_recommendations():
    """Show architecture recommendations."""
    from orchestrator.wordpress_plugin_rules import WordPressPluginRules
    
    rules = WordPressPluginRules()
    
    print("\n" + "=" * 70)
    print("Architecture Recommendations")
    print("=" * 70)
    
    scenarios = [
        {
            "name": "Public Plugin (WordPress.org)",
            "public": True,
            "team": 1,
            "complexity": "medium",
        },
        {
            "name": "Team Project (3 developers)",
            "public": False,
            "team": 3,
            "complexity": "complex",
        },
        {
            "name": "Rapid MVP (tight deadline)",
            "public": False,
            "team": 1,
            "complexity": "simple",
            "timeline": "tight",
        },
        {
            "name": "SaaS Integration",
            "public": False,
            "team": 2,
            "complexity": "complex",
        },
    ]
    
    for scenario in scenarios:
        recommended = rules.recommend_architecture_path(
            public_distribution=scenario.get("public", False),
            team_size=scenario.get("team", 1),
            complexity=scenario.get("complexity", "medium"),
            timeline=scenario.get("timeline", "normal"),
        )
        
        path_info = rules.get_architecture_path(recommended)
        
        print(f"\n📋 {scenario['name']}")
        print(f"   Public: {scenario['public']}, Team: {scenario['team']}, "
              f"Complexity: {scenario['complexity']}")
        print(f"   🎯 Recommended: {path_info['name']}")


def demo_generate_plugin(plugin_name: str):
    """Generate rules for a plugin."""
    from orchestrator.wordpress_plugin_rules import (
        WordPressPluginRules,
        generate_wordpress_plugin_rules,
    )
    
    print("=" * 70)
    print(f"Generating Rules: {plugin_name}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("./output/wordpress_plugin")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate rules
    rules_file = generate_wordpress_plugin_rules(
        plugin_name=plugin_name,
        output_dir=output_dir,
        public_distribution=True,
        team_size=2,
        complexity="medium",
        version="1.0.0",
        author="Your Name",
        include_tests=True,
        include_ci=True,
    )
    
    print(f"\n✅ Rules generated!")
    print(f"   File: {rules_file}")
    print(f"   Size: {rules_file.stat().st_size} bytes")
    
    # Show preview
    content = rules_file.read_text(encoding="utf-8")
    print(f"\n📄 Preview (first 50 lines):")
    print("-" * 70)
    for i, line in enumerate(content.split('\n')[:50], 1):
        print(f"{i:3}: {line}")
    print("-" * 70)
    print("... (truncated)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review {rules_file.name}")
    print(f"   2. Use these rules with the orchestrator")
    print(f"   3. Start building your plugin!")


def demo_comparison():
    """Compare architecture paths."""
    from orchestrator.wordpress_plugin_rules import WordPressPluginRules
    
    rules = WordPressPluginRules()
    
    print("\n" + "=" * 70)
    print("Architecture Path Comparison")
    print("=" * 70)
    
    print("\n| Feature | Modular OOP | Lightweight | Headless |")
    print("|---------|-------------|-------------|----------|")
    print("| Public Distribution | ✅ Best | ⚠️ Risky | ⚠️ Complex |")
    print("| Team Size > 1 | ✅ Best | ❌ Poor | ✅ Good |")
    print("| Complex Logic | ✅ Best | ❌ Poor | ✅ Good |")
    print("| Rapid MVP | ⚠️ Slow | ✅ Fast | ❌ Slow |")
    print("| Long-term Maint | ✅ Best | ❌ Poor | ✅ Good |")
    print("| Learning Curve | ⚠️ Steep | ✅ Easy | ⚠️ Steep |")
    print("| Scalability | ✅ High | ❌ Low | ✅ High |")
    print("| Test Coverage | ✅ Full | ⚠️ Limited | ✅ Good |")


def main():
    parser = argparse.ArgumentParser(description="WordPress Plugin Rules Demo")
    parser.add_argument("--demo", action="store_true", help="Run full demo")
    parser.add_argument("--generate", type=str, metavar="PLUGIN_NAME",
                       help="Generate rules for a plugin")
    parser.add_argument("--paths", action="store_true", help="Show architecture paths")
    parser.add_argument("--recommend", action="store_true", help="Show recommendations")
    parser.add_argument("--compare", action="store_true", help="Compare architectures")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_architecture_paths()
        demo_recommendations()
        demo_comparison()
    elif args.generate:
        demo_generate_plugin(args.generate)
    elif args.paths:
        demo_architecture_paths()
    elif args.recommend:
        demo_recommendations()
    elif args.compare:
        demo_comparison()
    else:
        print("WordPress Plugin Rules Examples")
        print("=" * 70)
        print()
        print("Available commands:")
        print("  --demo              Run full demonstration")
        print("  --generate NAME     Generate rules for plugin")
        print("  --paths             Show architecture paths")
        print("  --recommend         Show recommendations")
        print("  --compare           Compare architectures")
        print()
        print("Examples:")
        print('  python example_wordpress_plugin_rules.py --demo')
        print('  python example_wordpress_plugin_rules.py --generate "WooCommerce Enhancer"')


if __name__ == "__main__":
    main()
