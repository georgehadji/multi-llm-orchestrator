#!/usr/bin/env python3
"""Final cleanup of all temporary files."""
from pathlib import Path

files_to_remove = [
    # Cleanup scripts
    "cleanup_all.py",
    "cleanup_final.py",
    "cleanup_temp.py",
    "final_cleanup_all.py",
    # Any other temp files
    "OPTIMIZATION_REPORT.md.bak",
    "OPTIMIZATIONS_APPLIED.md.bak",
]

count = 0
for fname in files_to_remove:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")
        count += 1

# Remove __pycache__ directories
import shutil
for pycache in Path(".").rglob("__pycache__"):
    if pycache.is_dir():
        shutil.rmtree(pycache)
        print(f"✓ Removed: {pycache}")
        count += 1

if count == 0:
    print("No temporary files to clean")
else:
    print(f"\n✓ Cleaned {count} items")

# Self destruct
Path(__file__).unlink()
print("✓ Cleanup script removed itself")
