"""
Example: InDesign Plugin Rules
==============================
Demonstrates how to generate InDesign plugin development rules.

Usage:
    python example_indesign_plugin_rules.py --demo
    python example_indesign_plugin_rules.py --generate "My Plugin" --tech uxp
"""
import argparse
import sys
from pathlib import Path


def demo_technologies():
    """Show all technology paths."""
    from orchestrator.indesign_plugin_rules import InDesignPluginRules
    
    rules = InDesignPluginRules()
    
    print("=" * 70)
    print("InDesign Plugin Technology Paths")
    print("=" * 70)
    
    for key, path in rules.get_all_technologies().items():
        print(f"\n🔧 {path['name']}")
        print(f"   Key: {key}")
        print(f"   {path['description']}")
        print(f"\n   ✅ Pros:")
        for pro in path['pros']:
            print(f"      • {pro}")
        print(f"\n   ⚠️  Cons:")
        for con in path['cons']:
            print(f"      • {con}")
        print(f"\n   📋 Structure:")
        for item in path.get('structure', [])[:5]:
            print(f"      • {item}")


def demo_recommendations():
    """Show technology recommendations."""
    from orchestrator.indesign_plugin_rules import InDesignPluginRules
    
    rules = InDesignPluginRules()
    
    print("\n" + "=" * 70)
    print("Technology Recommendations")
    print("=" * 70)
    
    scenarios = [
        {
            "name": "Modern UI Panel",
            "native": False,
            "performance": False,
            "indesign": "2024",
        },
        {
            "name": "High-Performance Processor",
            "native": True,
            "performance": True,
            "hooks": True,
        },
        {
            "name": "Legacy Support (2019)",
            "native": False,
            "indesign": "2019",
        },
        {
            "name": "Custom Event Hooks",
            "native": True,
            "hooks": True,
        },
    ]
    
    for scenario in scenarios:
        recommended = rules.recommend_technology(
            requires_native_access=scenario.get("native", False),
            requires_high_performance=scenario.get("performance", False),
            requires_custom_hooks=scenario.get("hooks", False),
            target_indesign_version=scenario.get("indesign", "2024"),
        )
        
        path_info = rules.get_technology_path(recommended)
        
        print(f"\n📋 {scenario['name']}")
        print(f"   Requirements: Native={scenario.get('native')}, "
              f"Performance={scenario.get('performance')}")
        print(f"   🎯 Recommended: {path_info['name']}")


def demo_generate_plugin(plugin_name: str, technology: str):
    """Generate rules for a plugin."""
    from orchestrator.indesign_plugin_rules import (
        InDesignPluginRules,
        generate_indesign_plugin_rules,
    )
    
    print("=" * 70)
    print(f"Generating Rules: {plugin_name}")
    print(f"Technology: {technology.upper()}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("./output/indesign_plugin")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate rules
    use_ts = technology == "uxp"
    use_react = technology == "uxp"
    
    rules_file = generate_indesign_plugin_rules(
        plugin_name=plugin_name,
        output_dir=output_dir,
        technology=technology,
        use_typescript=use_ts,
        use_react=use_react,
        include_ci=True,
        gdpr_compliant=True,
    )
    
    print(f"\n✅ Rules generated!")
    print(f"   File: {rules_file}")
    print(f"   Size: {rules_file.stat().st_size} bytes")
    
    # Show preview
    content = rules_file.read_text(encoding="utf-8")
    print(f"\n📄 Preview (first 40 lines):")
    print("-" * 70)
    for i, line in enumerate(content.split('\n')[:40], 1):
        print(f"{i:3}: {line}")
    print("-" * 70)
    print("... (truncated)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review {rules_file.name}")
    print(f"   2. Setup your development environment")
    if technology == "uxp":
        print(f"   3. Run: npm init && npm install")
        print(f"   4. Start development with UXP Developer Tool")
    elif technology == "cpp":
        print(f"   3. Setup Visual Studio/Xcode with InDesign SDK")
        print(f"   4. Configure CMake build")


def demo_comparison():
    """Compare technologies."""
    print("\n" + "=" * 70)
    print("Technology Comparison")
    print("=" * 70)
    
    print("\n| Requirement | UXP | C++ | ExtendScript |")
    print("|-------------|-----|-----|--------------|")
    print("| Modern UI | ✅ | ⚠️ | ❌ |")
    print("| Cross-platform | ✅ | ❌ | ✅ |")
    print("| Deep native access | ⚠️ | ✅ | ❌ |")
    print("| High performance | ⚠️ | ✅ | ❌ |")
    print("| Rapid development | ✅ | ❌ | ✅ |")
    print("| Creative Cloud | ✅ | ⚠️ | ❌ |")
    print("| Legacy support | ❌ | ✅ | ✅ |")
    print("| Future-proof | ✅ | ✅ | ❌ |")


def main():
    parser = argparse.ArgumentParser(description="InDesign Plugin Rules Demo")
    parser.add_argument("--demo", action="store_true", help="Run full demo")
    parser.add_argument("--generate", type=str, metavar="PLUGIN_NAME",
                       help="Generate rules for a plugin")
    parser.add_argument("--tech", type=str, default="uxp",
                       choices=["uxp", "cpp", "extendscript"],
                       help="Technology choice")
    parser.add_argument("--technologies", action="store_true",
                       help="Show technology paths")
    parser.add_argument("--recommend", action="store_true",
                       help="Show recommendations")
    parser.add_argument("--compare", action="store_true",
                       help="Compare technologies")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_technologies()
        demo_recommendations()
        demo_comparison()
    elif args.generate:
        demo_generate_plugin(args.generate, args.tech)
    elif args.technologies:
        demo_technologies()
    elif args.recommend:
        demo_recommendations()
    elif args.compare:
        demo_comparison()
    else:
        print("InDesign Plugin Rules Examples")
        print("=" * 70)
        print()
        print("Available commands:")
        print("  --demo                  Run full demonstration")
        print("  --generate NAME         Generate rules for plugin")
        print("  --tech {uxp,cpp}        Technology choice")
        print("  --technologies          Show technology paths")
        print("  --recommend             Show recommendations")
        print("  --compare               Compare technologies")
        print()
        print("Examples:")
        print('  python example_indesign_plugin_rules.py --demo')
        print('  python example_indesign_plugin_rules.py --generate "Doc Processor" --tech uxp')
        print('  python example_indesign_plugin_rules.py --generate "Native Tool" --tech cpp')


if __name__ == "__main__":
    main()
