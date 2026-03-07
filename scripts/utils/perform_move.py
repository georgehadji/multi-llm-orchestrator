#!/usr/bin/env python3
"""Move test-related files from root to tests/ folder."""

import shutil
from pathlib import Path
import sys

# Root directory
root_dir = Path(__file__).parent
tests_dir = root_dir / "tests"

# List of files to move
files_to_move = [
    'test_ant_design_dashboard.py',
    'test_assembler_import.py',
    'test_deepseek.py',
    'test_enhanced_dashboard.py',
    'test_frontend_rules.py',
    'test_indesign_plugin_rules.py',
    'test_live_dashboard.py',
    'test_output_organizer.py',
    'test_performance_import.py',
    'test_wordpress_plugin_rules.py',
    'run_test.py',
    'check_frontend.py',
    'check_outputs.py',
    'check_state.py',
    'verify_frontend.py',
    'verify_syntax.py',
    'move_tests.py',
    'execute_move.py',
    'do_move.py',
    'run_move.py'
]

moved_count = 0
deleted_count = 0
errors = []

# Ensure tests directory exists
tests_dir.mkdir(exist_ok=True)

print("="*60)
print("FILE MOVE OPERATION - Root to tests/")
print("="*60)

for filename in files_to_move:
    source = root_dir / filename
    destination = tests_dir / filename
    
    if not source.exists():
        # File doesn't exist in root, skip
        continue
    
    if destination.exists():
        # Duplicate - delete from root
        try:
            source.unlink()
            print(f"[DELETED] {filename} (duplicate exists in tests/)")
            deleted_count += 1
        except Exception as e:
            error_msg = f"Error deleting {filename}: {e}"
            print(f"[ERROR] {error_msg}")
            errors.append(error_msg)
    else:
        # Move to tests folder
        try:
            shutil.move(str(source), str(destination))
            print(f"[MOVED]   {filename}")
            moved_count += 1
        except Exception as e:
            error_msg = f"Error moving {filename}: {e}"
            print(f"[ERROR]   {error_msg}")
            errors.append(error_msg)

print("="*60)
print("SUMMARY:")
print(f"  Files moved:   {moved_count}")
print(f"  Files deleted: {deleted_count} (duplicates)")
print(f"  Errors:        {len(errors)}")
if errors:
    for e in errors:
        print(f"    - {e}")
print("="*60)

# Self-destruct this script
script_path = Path(__file__)
try:
    script_path.unlink()
    print(f"\n[Cleaned up] {script_path.name}")
except Exception as e:
    print(f"\n[Warning] Could not delete {script_path.name}: {e}")

sys.exit(0 if not errors else 1)
