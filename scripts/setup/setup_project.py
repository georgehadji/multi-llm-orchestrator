#!/usr/bin/env python3
"""
Setup Project Structure
=======================
Creates scripts folder and organizes codebase.

Run: python setup_project.py
"""
import os
import shutil
from pathlib import Path

def create_scripts_folder():
    """Create scripts folder with utility scripts."""
    project_root = Path(__file__).parent
    scripts_dir = project_root / "scripts"
    
    # Create directory
    scripts_dir.mkdir(exist_ok=True)
    print(f"✅ Created: {scripts_dir}")
    
    # Create __init__.py
    (scripts_dir / "__init__.py").write_text('"""Scripts Package for Multi-LLM Orchestrator"""\n')
    
    # Script definitions
    scripts = {
        "run_dashboard.py": '''#!/usr/bin/env python3
"""Run the Ant Design dashboard."""
import argparse
from orchestrator.dashboard_antd import run_ant_design_dashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mission Control Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()
    
    run_ant_design_dashboard(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )
''',
        "run_tests.py": '''#!/usr/bin/env python3
"""Run all tests with coverage."""
import subprocess
import sys

cmd = [
    sys.executable, "-m", "pytest",
    "tests/", "-v",
    "--cov=orchestrator",
    "--cov-report=html:htmlcov",
    "--cov-report=term-missing"
]
result = subprocess.run(cmd)
sys.exit(result.returncode)
''',
        "organize_output.py": '''#!/usr/bin/env python3
"""Organize project output."""
import asyncio
import sys
from pathlib import Path
from orchestrator.output_organizer import organize_project_output

async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/organize_output.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    print(f"Organizing: {output_dir}")
    report = await organize_project_output(output_dir)
    
    print(f"\\n✅ Tasks moved: {len(report.tasks_moved)}")
    print(f"✅ Tests generated: {len(report.tests_created)}")
    if report.tests_run:
        passed = sum(1 for r in report.tests_run if r.passed)
        print(f"✅ Tests passed: {passed}/{len(report.tests_run)}")

if __name__ == "__main__":
    asyncio.run(main())
''',
        "check_models.py": '''#!/usr/bin/env python3
"""Check model availability and health."""
from orchestrator.models import Model, get_provider
from orchestrator.api_clients import UnifiedClient

def main():
    client = UnifiedClient()
    
    print("=" * 70)
    print("Model Availability Check")
    print("=" * 70)
    print(f"{'Model':<25} | {'Provider':<12} | {'Status':<15} | {'Reason'}")
    print("-" * 70)
    
    for model in Model:
        available = client.is_available(model)
        provider = get_provider(model)
        status = "✅ Available" if available else "❌ Unavailable"
        reason = "API key configured" if available else "API key missing"
        
        print(f"{model.value:<25} | {provider:<12} | {status:<15} | {reason}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
''',
        "cleanup_cache.py": '''#!/usr/bin/env python3
"""Clean up cache files."""
import shutil
from pathlib import Path

def main():
    cache_dirs = [
        Path.home() / ".orchestrator_cache",
        Path("__pycache__"),
        Path(".pytest_cache"),
        Path(".ruff_cache"),
    ]
    
    print("Cleaning up cache files...")
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  Removed: {cache_dir}")
    
    print("\\n✅ Cleanup complete")

if __name__ == "__main__":
    main()
''',
        "create_project.py": '''#!/usr/bin/env python3
"""Create a new project from CLI."""
import asyncio
import argparse
from pathlib import Path
from orchestrator import Orchestrator, Budget

async def main():
    parser = argparse.ArgumentParser(description="Create a new project")
    parser.add_argument("--project", "-p", required=True, help="Project description")
    parser.add_argument("--criteria", "-c", required=True, help="Success criteria")
    parser.add_argument("--budget", "-b", type=float, default=5.0, help="Budget in USD")
    parser.add_argument("--time", "-t", type=int, default=3600, help="Time limit in seconds")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory")
    args = parser.parse_args()
    
    orch = Orchestrator(budget=Budget(max_usd=args.budget, max_time_seconds=args.time))
    
    print(f"Starting project: {args.project[:50]}...")
    print(f"Budget: ${args.budget}, Time: {args.time}s")
    print("-" * 50)
    
    state = await orch.run_project(
        project_description=args.project,
        success_criteria=args.criteria,
        output_dir=Path(args.output)
    )
    
    print(f"\\nStatus: {state.status.value}")
    print(f"Cost: ${state.budget.spent_usd:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
''',
    }
    
    for name, content in scripts.items():
        script_path = scripts_dir / name
        script_path.write_text(content)
        # Try to make executable (Unix only)
        try:
            os.chmod(script_path, 0o755)
        except:
            pass
        print(f"✅ Created: scripts/{name}")
    
    print(f"\\n🎉 Scripts folder ready!")
    print("\\nAvailable commands:")
    for name in scripts.keys():
        print(f"  python scripts/{name}")

def organize_root_files():
    """Move loose files to appropriate locations."""
    project_root = Path(__file__).parent
    
    # Files to keep in root
    keep_in_root = [
        ".env", ".env.example", ".gitignore", "LICENSE",
        "pyproject.toml", "setup.py", "Makefile", "README.md",
        "CLAUDE.md", "USAGE_GUIDE.md", "CAPABILITIES.md",
        "setup_project.py", "check_outputs.py", "test_deepseek.py"
    ]
    
    # Example files to move
    example_files = [
        "example_enhanced_dashboard.py",
        "example_deepseek_coder_v2.py",
        "example_architecture_rules.py"
    ]
    
    # Move example files
    examples_dir = project_root / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    for file in example_files:
        src = project_root / file
        if src.exists():
            dst = examples_dir / file
            shutil.move(str(src), str(dst))
            print(f"📁 Moved: {file} → examples/")
    
    print("✅ Root files organized")

def main():
    print("=" * 70)
    print("Setting up Multi-LLM Orchestrator Project")
    print("=" * 70)
    
    create_scripts_folder()
    organize_root_files()
    
    print("\\n" + "=" * 70)
    print("✅ Setup complete!")
    print("=" * 70)
    print("\\nNext steps:")
    print("  1. Start dashboard: python scripts/run_dashboard.py")
    print("  2. Check models:   python scripts/check_models.py")
    print("  3. Create project: python scripts/create_project.py -p \"Build API\" -c \"Tests pass\"")

if __name__ == "__main__":
    main()
