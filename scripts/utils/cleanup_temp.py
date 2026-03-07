"""Clean up temporary files."""
import os
from pathlib import Path

files_to_remove = [
    "create_github_workflow.py",
    "create_workflow.py",
    "test_assembler_import.py",
]

for fname in files_to_remove:
    fpath = Path(fname)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fname}")
    else:
        print(f"○ Not found: {fname}")

# Remove self
Path(__file__).unlink()
print("✓ Removed: cleanup_temp.py")
