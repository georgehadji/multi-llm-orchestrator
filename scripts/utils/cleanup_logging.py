#!/usr/bin/env python3
"""Delete the problematic logging.py file"""
import os
import sys

filepath = os.path.join(os.path.dirname(__file__), 'orchestrator', 'logging.py')

if os.path.exists(filepath):
    try:
        os.remove(filepath)
        print(f"✅ Deleted: {filepath}")
    except Exception as e:
        print(f"❌ Error deleting file: {e}")
        sys.exit(1)
else:
    print(f"⏩ File does not exist: {filepath}")

print("\nCleanup complete!")
