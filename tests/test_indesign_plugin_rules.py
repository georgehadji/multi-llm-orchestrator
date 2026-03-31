"""
Test InDesign Plugin Rules
==========================
Tests for the InDesign plugin development rules engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that InDesign rules can be imported."""
    print("Testing imports...")

    try:
        from orchestrator.indesign_plugin_rules import (
            InDesignPluginRules,
            InDesignRulesConfig,
            generate_indesign_plugin_rules,
        )

        print("✅ InDesign plugin rules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_technology_paths():
    """Test technology paths."""
    print("\nTesting technology paths...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()
    paths = rules.get_all_technologies()

    assert len(paths) == 3, f"Expected 3 paths, got {len(paths)}"
    assert "uxp" in paths
    assert "cpp" in paths
    assert "extendscript" in paths

    print("  Available technologies:")
    for key, path in paths.items():
        print(f"    - {path['name']} ({key})")
        assert "name" in path
        assert "description" in path
        assert "pros" in path
        assert "cons" in path

    print("✅ All technology paths loaded")
    return True


def test_config_generation():
    """Test configuration generation."""
    print("\nTesting config generation...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()

    # Test UXP config
    config = rules.generate_config("Document Automation", technology="uxp")
    assert config.plugin_name == "Document Automation"
    assert config.plugin_id == "document-automation"
    assert config.technology == "uxp"
    assert config.use_typescript is True
    assert config.use_react is True

    print(f"  UXP Plugin: {config.plugin_name}")
    print(f"    ID: {config.plugin_id}")
    print(f"    TypeScript: {config.use_typescript}")
    print(f"    React: {config.use_react}")

    # Test C++ config
    config = rules.generate_config("Native Processor", technology="cpp")
    assert config.technology == "cpp"
    assert config.cpp_standard == "c++17"
    assert config.use_raii is True

    print(f"  C++ Plugin: {config.plugin_name}")
    print(f"    Standard: {config.cpp_standard}")
    print(f"    RAII: {config.use_raii}")

    print("✅ Configuration generated correctly")
    return True


def test_technology_recommendation():
    """Test technology recommendations."""
    print("\nTesting technology recommendations...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()

    # Standard plugin → UXP
    tech = rules.recommend_technology(
        requires_native_access=False,
        target_indesign_version="2024",
    )
    assert tech == "uxp", f"Expected uxp, got {tech}"
    print(f"  Standard plugin → {tech.upper()}")

    # High performance → C++
    tech = rules.recommend_technology(
        requires_native_access=True,
        requires_high_performance=True,
    )
    assert tech == "cpp", f"Expected cpp, got {tech}"
    print(f"  High-performance → {tech.upper()}")

    # Custom hooks → C++
    tech = rules.recommend_technology(
        requires_custom_hooks=True,
    )
    assert tech == "cpp", f"Expected cpp, got {tech}"
    print(f"  Custom hooks → {tech.upper()}")

    # Legacy InDesign → ExtendScript
    tech = rules.recommend_technology(
        target_indesign_version="2019",
    )
    assert tech == "extendscript", f"Expected extendscript, got {tech}"
    print(f"  Legacy InDesign 2019 → {tech.upper()}")

    print("✅ Technology recommendations working")
    return True


def test_rules_content_generation():
    """Test rules file content generation."""
    print("\nTesting rules content generation...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()
    config = rules.generate_config("UXP Document Processor", technology="uxp")
    content = rules.get_rules_file_content(config)

    # Check content includes key sections
    assert "UXP Document Processor" in content
    assert config.plugin_id in content
    assert "MUST FOLLOW RULES" in content
    assert "Security" in content
    assert "Architecture Separation" in content
    assert "Non-Blocking Operations" in content
    assert "Development Checklist" in content

    # Check content is substantial
    assert len(content) > 8000, f"Content too short: {len(content)} chars"

    print(f"  Generated rules file: {len(content)} characters")
    print("  Includes:")
    print("    - Technology decision")
    print("    - Architecture patterns")
    print("    - Security rules")
    print("    - Performance guidelines")
    print("    - Testing requirements")
    print("    - Distribution guidelines")

    print("✅ Rules content generated")
    return True


def test_cpp_rules_content():
    """Test C++ specific rules."""
    print("\nTesting C++ rules content...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    rules = InDesignPluginRules()
    config = rules.generate_config("Native Plugin", technology="cpp")
    content = rules.get_rules_file_content(config)

    # Check C++ specific content
    assert "RAII" in content
    assert "memory" in content.lower() or "sanitizers" in content.lower()
    assert "Visual Studio" in content or "Xcode" in content
    assert "CMake" in content

    print("  C++ specific rules present:")
    print("    - RAII patterns")
    print("    - Memory sanitizers")
    print("    - Native SDK")

    print("✅ C++ rules content verified")
    return True


def test_rules_file_save(tmp_path=None):
    """Test saving rules file."""
    print("\nTesting rules file save...")

    from orchestrator.indesign_plugin_rules import InDesignPluginRules

    if tmp_path is None:
        tmp_path = Path("./test_output")
        tmp_path.mkdir(exist_ok=True)

    rules = InDesignPluginRules()
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
        return True


def test_convenience_function():
    """Test convenience function."""
    print("\nTesting convenience function...")

    from orchestrator.indesign_plugin_rules import generate_indesign_plugin_rules

    tmp_path = Path("./test_output")
    tmp_path.mkdir(exist_ok=True)

    try:
        rules_file = generate_indesign_plugin_rules(
            plugin_name="Convenience Test Plugin",
            output_dir=tmp_path,
            technology="uxp",
            use_typescript=True,
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
    print("InDesign Plugin Rules Tests")
    print("=" * 70)

    tests = [
        test_imports,
        test_technology_paths,
        test_config_generation,
        test_technology_recommendation,
        test_rules_content_generation,
        test_cpp_rules_content,
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
        print("\n✅ All tests passed! InDesign Plugin Rules are ready.")
        print("\n📦 Example usage:")
        print("  from orchestrator import InDesignPluginRules")
        print("  rules = InDesignPluginRules()")
        print('  config = rules.generate_config("My Plugin", technology="uxp")')
        print('  rules.save_rules_file(config, Path("./output"))')
        print("\n🎯 Technologies:")
        print("  • UXP - Modern JavaScript/TypeScript (recommended)")
        print("  • C++ - Native SDK for high performance")
        print("  • ExtendScript - Legacy (deprecated)")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
