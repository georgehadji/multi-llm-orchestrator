#!/usr/bin/env python3
"""Clean Python cache - Windows compatible."""
import os
import shutil
from pathlib import Path

def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal."""
    os.chmod(path, 0o777)
    func(path)

def clean():
    root = Path(".")
    count = 0
    
    for pycache in list(root.rglob("__pycache__")):
        if pycache.is_dir():
            try:
                shutil.rmtree(pycache, onerror=remove_readonly)
                count += 1
                print(f"Removed: {pycache}")
            except Exception as e:
                print(f"Could not remove {pycache}: {e}")
    
    for pyc in list(root.rglob("*.pyc")):
        try:
            os.chmod(pyc, 0o777)
            pyc.unlink()
            print(f"Removed: {pyc}")
        except Exception as e:
            print(f"Could not remove {pyc}: {e}")
    
    print(f"\nCleaned {count} __pycache__ directories")

if __name__ == "__main__":
    clean()
