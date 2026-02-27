#!/usr/bin/env python3
"""Clean up all temporary files."""
from pathlib import Path

temp_files = [
    # Cleanup scripts
    "cleanup_final.py",
    "cleanup_temp.py",
    # Create scripts
    "create_all_files.py",
    "create_configs.py",
    "create_github_templates.py",
    "create_github_workflow.py",
    "create_scripts.py",
    "create_workflow.py",
    "init_project_structure.py",
]

count = 0
for fname in temp_files:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")
        count += 1

if count == 0:
    print("No temporary files to clean")
else:
    print(f"\n✓ Cleaned {count} temporary files")

# Self destruct
Path(__file__).unlink()
