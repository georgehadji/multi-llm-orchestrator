#!/usr/bin/env python3
"""
Fix git errors - Remove problematic files
"""
import os
from pathlib import Path

print("🔧 Fixing Git Errors...")
print("=" * 50)

# Files to remove
problematic_files = [
    "nul",           # Reserved Windows filename
    "h origin main", # Accidental file from command
]

removed = []
for filename in problematic_files:
    filepath = Path(filename)
    if filepath.exists():
        try:
            filepath.unlink()
            removed.append(filename)
            print(f"✓ Removed: {filename}")
        except Exception as e:
            print(f"✗ Could not remove {filename}: {e}")
    else:
        print(f"✓ Already gone: {filename}")

print("\n" + "=" * 50)
print("✅ Cleanup complete!")
print("\nNow run:")
print("  git add -A")
print("  git commit -m \"feat: v5.1 Management Systems + v5.0 Performance Optimization\"")
print("  git push origin release/v5.1")
