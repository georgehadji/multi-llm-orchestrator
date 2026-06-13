#!/usr/bin/env python3
"""
Convert a root-level duplicate module to a backward-compatibility shim.

Usage:
    python scripts/convert_to_shim.py <root_file> <target_import>

Example:
    python scripts/convert_to_shim.py orchestrator/async_file_io.py events.async_file_io
"""
import argparse
import sys
from pathlib import Path


SHIM_TEMPLATE = '''"""
{module_name} — Backward-compatibility shim
The canonical implementation lives in {canonical_location}.
New code should import from `{new_import_path}` directly.
"""

from {import_stmt} import *  # noqa: F401, F403
'''


def convert_to_shim(root_file: Path, target_import: str) -> None:
    """Convert a root file to a shim re-exporting from target."""
    
    if not root_file.exists():
        print(f"Error: {root_file} does not exist")
        sys.exit(1)
    
    module_name = root_file.stem
    
    # Build import statement
    if "." in target_import:
        # Relative import: orchestrator.events.async_file_io -> .events.async_file_io
        import_stmt = "." + target_import.replace("orchestrator.", "", 1)
        canonical_location = target_import
        new_import_path = target_import
    else:
        import_stmt = f".{target_import}"
        canonical_location = f"orchestrator.{target_import}"
        new_import_path = canonical_location
    
    # Create shim content
    shim_content = SHIM_TEMPLATE.format(
        module_name=module_name,
        canonical_location=canonical_location,
        new_import_path=new_import_path,
        import_stmt=import_stmt,
    )
    
    # Backup original
    backup_file = root_file.with_suffix(".py.bak")
    original_content = root_file.read_text(encoding="utf-8")
    backup_file.write_text(original_content, encoding="utf-8")
    
    # Write shim
    root_file.write_text(shim_content, encoding="utf-8")
    
    print(f"Converted: {root_file}")
    print(f"  Backup: {backup_file}")
    print(f"  Now re-exports from: {import_stmt}")
    print(f"  Reduced from {len(original_content.splitlines())} lines to {len(shim_content.splitlines())} lines")


def main():
    parser = argparse.ArgumentParser(description="Convert duplicate module to shim")
    parser.add_argument("root_file", help="Path to root-level file to convert")
    parser.add_argument("target_import", help="Target import path (e.g., 'events.async_file_io')")
    
    args = parser.parse_args()
    
    root_file = Path(args.root_file)
    convert_to_shim(root_file, args.target_import)


if __name__ == "__main__":
    main()
