#!/usr/bin/env python3
"""Clean up temporary files."""
from pathlib import Path
import shutil

temp_files = [
    "cleanup_temp.py",
    "cleanup_final.py",
    "cleanup_all.py",
    "final_cleanup_all.py",
    "remove_all_temp.py",
    "create_all_files.py",
    "create_configs.py",
    "create_github_templates.py",
    "create_github_workflow.py",
    "create_scripts.py",
    "create_workflow.py",
    "init_project_structure.py",
    "cleanup_temp_files.py",
]

count = 0
for fname in temp_files:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")
        count += 1

# Clean __pycache__ directories
for pycache in list(Path(".").rglob("__pycache__")):
    if pycache.is_dir():
        shutil.rmtree(pycache)
        count += 1

print(f"\n✅ Cleaned {count} items")
