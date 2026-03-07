#!/usr/bin/env python3
"""
Setup Scripts - Creates scripts folder structure
"""
import os
from pathlib import Path

# Create scripts folder
scripts_dir = Path("scripts")
scripts_dir.mkdir(exist_ok=True)

# Create __init__.py
(scripts_dir / "__init__.py").write_text('"""Scripts Package"""\n')

# Create common utility scripts
scripts = {
    "run_dashboard.py": '''#!/usr/bin/env python3
"""Run the enhanced dashboard."""
from orchestrator.dashboard_enhanced import run_enhanced_dashboard

if __name__ == "__main__":
    run_enhanced_dashboard()
''',
    "run_tests.py": '''#!/usr/bin/env python3
"""Run all tests with coverage."""
import subprocess
import sys

cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=orchestrator", "--cov-report=html"]
subprocess.run(cmd)
''',
    "organize_output.py": '''#!/usr/bin/env python3
"""Organize project output."""
import asyncio
import sys
from pathlib import Path
from orchestrator.output_organizer import organize_project_output

async def main():
    if len(sys.argv) < 2:
        print("Usage: python organize_output.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    report = await organize_project_output(output_dir)
    print(f"Organized: {len(report.tasks_moved)} tasks, {len(report.tests_run)} tests")

if __name__ == "__main__":
    asyncio.run(main())
''',
    "check_models.py": '''#!/usr/bin/env python3
"""Check model availability and health."""
from orchestrator import Model, get_provider
from orchestrator.api_clients import UnifiedClient

def main():
    client = UnifiedClient()
    print("=" * 60)
    print("Model Availability Check")
    print("=" * 60)
    
    for model in Model:
        available = client.is_available(model)
        provider = get_provider(model)
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{model.value:<25} | {provider:<12} | {status}")

if __name__ == "__main__":
    main()
''',
}

for name, content in scripts.items():
    (scripts_dir / name).write_text(content)
    print(f"Created: scripts/{name}")

print("\n✅ Scripts folder setup complete!")
