#!/usr/bin/env python3
"""Clear Python cache files"""
import os
import shutil

count = 0
for root, dirs, files in os.walk('.'):
    for d in dirs:
        if d == '__pycache__':
            path = os.path.join(root, d)
            try:
                shutil.rmtree(path)
                count += 1
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")

print(f"\n✅ Cleared {count} cache directories")
