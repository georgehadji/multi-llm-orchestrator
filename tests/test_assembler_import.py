"""Test that ProjectAssembler module loads correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

try:
    from orchestrator.project_assembler import ProjectAssembler, DependencyAnalyzer, ModuleInfo

    print("✓ ProjectAssembler imports successfully")
    print(f"  - ProjectAssembler: {ProjectAssembler}")
    print(f"  - DependencyAnalyzer: {DependencyAnalyzer}")
    print(f"  - ModuleInfo: {ModuleInfo}")

    # Check methods
    print("\n✓ ProjectAssembler methods:")
    for method in ["assemble", "_generate_directory_structure", "_generate_pyproject_toml"]:
        if hasattr(ProjectAssembler, method):
            print(f"  - {method}")

    # Check DependencyAnalyzer methods
    print("\n✓ DependencyAnalyzer methods:")
    for method in ["extract_imports", "extract_exports"]:
        if hasattr(DependencyAnalyzer, method):
            print(f"  - {method}")

    print("\n✓ All imports successful!")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
