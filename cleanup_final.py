#!/usr/bin/env python3
"""Final cleanup of temporary files."""
from pathlib import Path

files_to_remove = [
    "create_all_files.py",
    "create_configs.py",
    "create_github_templates.py",
    "create_github_workflow.py",
    "create_scripts.py",
    "create_workflow.py",
    "init_project_structure.py",
]

for fname in files_to_remove:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")

print("\n✓ Cleanup complete!")

# Self destruct
Path(__file__).unlink()
