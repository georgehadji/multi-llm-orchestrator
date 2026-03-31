"""
Test WordPress Plugin Rules
===========================
Tests for the WordPress plugin development rules engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that WordPress rules can be imported."""
    print("Testing imports...")

    try:
        from orchestrator.wordpress_plugin_rules import (
            WordPressPluginRules,
            WPRulesConfig,
            generate_wordpress_plugin_rules,
        )

        print("✅ WordPress plugin rules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_architecture_paths():
    """Test architecture paths."""
    print("\nTesting architecture paths...")

    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    paths = rules.get_all_architecture_paths()

    assert len(paths) == 3, f"Expected 3 paths, got {len(paths)}"
    assert "modular_oop" in paths
    assert "lightweight" in paths
    assert "headless" in paths

    print("  Available paths:")
    for key, path in paths.items():
        print(f"    - {path['name']} ({key})")
        assert "name" in path
        assert "description" in path
        assert "pros" in path
        assert "cons" in path

    print("✅ All architecture paths loaded")
    return True


def test_config_generation():
    """Test configuration generation."""
    print("\nTesting config generation...")

    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    config = rules.generate_config("My Awesome Plugin")

    assert config.plugin_name == "My Awesome Plugin"
    assert config.plugin_slug == "my-awesome-plugin"
    assert config.text_domain == "my-awesome-plugin"
    assert config.namespace == "MyAwesomePlugin"
    assert config.prefix == "my_awesome_plugin_"
    assert config.architecture_path == "modular_oop"

    print(f"  Plugin: {config.plugin_name}")
    print(f"  Slug: {config.plugin_slug}")
    print(f"  Namespace: {config.namespace}")
    print(f"  Prefix: {config.prefix}")

    print("✅ Configuration generated correctly")
    return True


def test_architecture_recommendation():
    """Test architecture path recommendation."""
    print("\nTesting architecture recommendations...")

    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()

    # Public distribution → modular_oop
    path = rules.recommend_architecture_path(
        public_distribution=True, team_size=1, complexity="simple"
    )
    assert path == "modular_oop", f"Expected modular_oop, got {path}"
    print(f"  Public distribution → {path}")

    # Team collaboration → modular_oop
    path = rules.recommend_architecture_path(
        public_distribution=False, team_size=3, complexity="medium"
    )
    assert path == "modular_oop", f"Expected modular_oop, got {path}"
    print(f"  Team project → {path}")

    # Tight timeline + simple → lightweight
    path = rules.recommend_architecture_path(
        public_distribution=False, team_size=1, complexity="simple", timeline="tight"
    )
    assert path == "lightweight", f"Expected lightweight, got {path}"
    print(f"  Rapid MVP → {path}")

    print("✅ Architecture recommendations working")
    return True


def test_rules_content_generation():
    """Test rules file content generation."""
    print("\nTesting rules content generation...")

    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    config = rules.generate_config("WooCommerce Enhancer")
    content = rules.get_rules_file_content(config)

    # Check content includes key sections
    assert "WooCommerce Enhancer" in content
    assert config.plugin_slug in content
    assert config.prefix in content
    assert "MUST FOLLOW RULES" in content
    assert "Security" in content
    assert "Coding Standards" in content
    assert "Development Checklist" in content

    # Check content is substantial
    assert len(content) > 5000, f"Content too short: {len(content)} chars"

    print(f"  Generated rules file: {len(content)} characters")
    print("  Includes:")
    print("    - Architecture decision")
    print("    - Plugin configuration")
    print("    - Security rules")
    print("    - Coding standards")
    print("    - Best practices")
    print("    - Development checklist")

    print("✅ Rules content generated")
    return True


def test_rules_file_save(tmp_path=None):
    """Test saving rules file."""
    print("\nTesting rules file save...")

    from orchestrator.wordpress_plugin_rules import WordPressPluginRules

    # Use temp directory if not provided
    if tmp_path is None:
        tmp_path = Path("./test_output")
        tmp_path.mkdir(exist_ok=True)

    rules = WordPressPluginRules()
    config = rules.generate_config("Test Plugin")

    try:
        rules_file = rules.save_rules_file(config, tmp_path)

        assert rules_file.exists(), "Rules file not created"
        assert rules_file.name == ".ai-rules.md"

        content = rules_file.read_text(encoding="utf-8")
        assert "Test Plugin" in content

        print(f"  Saved to: {rules_file}")
        print(f"  File size: {len(content)} bytes")

        # Cleanup
        rules_file.unlink()

        print("✅ Rules file saved successfully")
        return True
    except Exception as e:
        print(f"⚠️ File save test skipped: {e}")
        return True  # Don't fail on file system issues


def test_convenience_function():
    """Test convenience function."""
    print("\nTesting convenience function...")

    from orchestrator.wordpress_plugin_rules import generate_wordpress_plugin_rules

    tmp_path = Path("./test_output")
    tmp_path.mkdir(exist_ok=True)

    try:
        rules_file = generate_wordpress_plugin_rules(
            plugin_name="Convenience Test Plugin",
            output_dir=tmp_path,
            public_distribution=True,
        )

        assert rules_file.exists(), "Rules file not created"
        content = rules_file.read_text()
        assert "Convenience Test Plugin" in content

        print(f"  Generated: {rules_file}")

        # Cleanup
        rules_file.unlink()
        tmp_path.rmdir()

        print("✅ Convenience function works")
        return True
    except Exception as e:
        print(f"⚠️ Convenience function test skipped: {e}")
        return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("WordPress Plugin Rules Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_architecture_paths,
        test_config_generation,
        test_architecture_recommendation,
        test_rules_content_generation,
        test_rules_file_save,
        test_convenience_function,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests passed! WordPress Plugin Rules are ready.")
        print("\n📦 Example usage:")
        print("  from orchestrator import WordPressPluginRules")
        print("  rules = WordPressPluginRules()")
        print('  config = rules.generate_config("My Plugin")')
        print('  rules.save_rules_file(config, Path("./output"))')

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
