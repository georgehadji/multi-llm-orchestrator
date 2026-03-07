#!/usr/bin/env python3
"""Remove all temporary files."""
from pathlib import Path
import shutil

temp_files = [
    # Create scripts
    "create_all_files.py",
    "create_configs.py",
    "create_github_templates.py",
    "create_github_workflow.py",
    "create_scripts.py",
    "create_workflow.py",
    "init_project_structure.py",
    # Cleanup scripts
    "cleanup_all.py",
    "cleanup_final.py",
    "cleanup_temp.py",
    "final_cleanup_all.py",
    # This script
    "remove_all_temp.py",
]

count = 0
for fname in temp_files:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")
        count += 1

# Clean __pycache__ directories
for pycache in Path(".").rglob("__pycache__"):
    if pycache.is_dir():
        shutil.rmtree(pycache)
        print(f"✓ Removed: {pycache}")
        count += 1

print(f"\n✅ Cleaned {count} items")
