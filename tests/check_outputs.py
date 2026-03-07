#!/usr/bin/env python3
"""Check output files."""
from pathlib import Path

outputs_dir = Path("outputs")
print("Files in outputs/:")
for f in sorted(outputs_dir.glob("*")):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name:40} {size:>10,} bytes")
