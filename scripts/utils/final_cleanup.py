"""Final cleanup of temporary files."""
from pathlib import Path
import shutil

files_to_remove = [
    "orchestrator/output_writer_trimmed.py",
    "cleanup_temp.py",
]

for fpath_str in files_to_remove:
    fpath = Path(fpath_str)
    if fpath.exists():
        fpath.unlink()
        print(f"✓ Removed: {fpath_str}")

# Clean __pycache__ directories
for cache_dir in Path(".").rglob("__pycache__"):
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        print(f"✓ Removed: {cache_dir}")

# Self destruct
Path(__file__).unlink()
print("✓ Removed: final_cleanup.py")
print("\n✅ Cleanup complete!")
