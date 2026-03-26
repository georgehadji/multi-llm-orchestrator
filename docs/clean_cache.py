#!/usr/bin/env python3
"""Clean Python cache files."""
import shutil
import os
from pathlib import Path

def clean_cache():
    root = Path(".")
    removed_dirs = 0
    removed_files = 0
    
    # Remove __pycache__ directories
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            try:
                shutil.rmtree(pycache)
                removed_dirs += 1
                print(f"Removed: {pycache}")
            except Exception as e:
                print(f"Error removing {pycache}: {e}")
    
    # Remove .pyc files
    for pyc in root.rglob("*.pyc"):
        try:
            pyc.unlink()
            removed_files += 1
            print(f"Removed: {pyc}")
        except Exception as e:
            print(f"Error removing {pyc}: {e}")
    
    print(f"\nCleaned: {removed_dirs} directories, {removed_files} files")

if __name__ == "__main__":
    clean_cache()
