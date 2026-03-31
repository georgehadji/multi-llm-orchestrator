"""
Test script για το Architecture Optimization feature.

Τρέξε με:
    python test_architecture_optimization.py
"""

import asyncio
from pathlib import Path

# Test 1: Basic imports
print("=" * 60)
print("Test 1: Basic Imports")
print("=" * 60)
try:
    from orchestrator.architecture_rules import (
        ProjectRules,
        ArchitectureRulesEngine,
        RulesGenerator,
        ArchitectureDecision,
        TechnologyStack,
        CodingStandard,
        ArchitecturalStyle,
        ProgrammingParadigm,
        APIStyle,
        DatabaseType,
    )

    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: ProjectRules with metadata
print("\n" + "=" * 60)
print("Test 2: ProjectRules with Metadata")
print("=" * 60)
try:
    rules = ProjectRules(project_type="test", _llm_generated=True, _llm_optimized=False)
    print(f"✅ ProjectRules created")
    print(f"   _llm_generated: {rules._llm_generated}")
    print(f"   _llm_optimized: {rules._llm_optimized}")

    # Test to_yaml excludes metadata
    yaml_str = rules.to_yaml()
    if "_llm_generated" not in yaml_str and "_llm_optimized" not in yaml_str:
        print("✅ Metadata correctly excluded from YAML")
    else:
        print("⚠️  Warning: Metadata found in YAML")
except Exception as e:
    print(f"❌ Test failed: {e}")

# Test 3: Rule-based generation
print("\n" + "=" * 60)
print("Test 3: Rule-based Generation")
print("=" * 60)
try:
    generator = RulesGenerator()
    rules = generator.generate_rules(
        description="Build a REST API with Python and FastAPI",
        criteria="High performance, scalable",
        project_type="web_api",
    )
    print(f"✅ Rules generated")
    print(f"   Style: {rules.architecture.style.value}")
    print(f"   Paradigm: {rules.architecture.paradigm.value}")
    print(f"   Stack: {rules.architecture.stack.primary_language}")
except Exception as e:
    print(f"❌ Test failed: {e}")

# Test 4: ArchitectureRulesEngine without client (rule-based only)
print("\n" + "=" * 60)
print("Test 4: ArchitectureRulesEngine (Rule-based)")
print("=" * 60)


def test_engine_sync():
    engine = ArchitectureRulesEngine(client=None)

    async def async_test():
        rules = await engine.generate_rules(
            description="Build a React frontend dashboard",
            criteria="Responsive, modern UI",
            project_type="web_frontend",
        )
        print(f"✅ Rules generated")
        print(f"   Style: {rules.architecture.style.value}")
        print(f"   Stack: {rules.architecture.stack.primary_language}")
        print(f"   _llm_generated: {rules._llm_generated}")
        print(f"   _llm_optimized: {rules._llm_optimized}")

        # Test summary generation
        summary = engine.generate_summary(rules)
        print("\n📋 Generated Summary:")
        print(summary)

    asyncio.run(async_test())


try:
    test_engine_sync()
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Optimization prompt structure
print("\n" + "=" * 60)
print("Test 5: Optimization Method Exists")
print("=" * 60)
try:
    engine = ArchitectureRulesEngine(client=None)
    assert hasattr(engine, "_optimize_rules_with_llm"), "Method not found"
    print("✅ _optimize_rules_with_llm method exists")
except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
