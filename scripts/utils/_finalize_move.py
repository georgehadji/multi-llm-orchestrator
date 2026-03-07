#!/usr/bin/env python3
"""Move all test-related Python files from root to tests/ folder.

This script handles all 20 files specified by the user:
- Moves files that don't exist in tests/
- Deletes duplicates from root if they already exist in tests/
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator")
TESTS_DIR = ROOT_DIR / "tests"

FILES_TO_MOVE = [
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
    'run_move.py',
]


def main():
    TESTS_DIR.mkdir(exist_ok=True)
    
    moved = 0
    deleted = 0
    not_found = 0
    errors = []
    
    print("=" * 60)
    print("TEST FILES MIGRATION - Root to tests/")
    print("=" * 60)
    
    for filename in FILES_TO_MOVE:
        src = ROOT_DIR / filename
        dst = TESTS_DIR / filename
        
        if not src.exists():
            not_found += 1
            continue
        
        try:
            if dst.exists():
                # Duplicate - delete from root
                src.unlink()
                print(f"[DELETED] {filename} (duplicate in tests/)")
                deleted += 1
            else:
                # Move to tests folder
                shutil.move(str(src), str(dst))
                print(f"[MOVED]   {filename}")
                moved += 1
        except Exception as e:
            print(f"[ERROR]   {filename}: {e}")
            errors.append((filename, str(e)))
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"  Files moved:              {moved}")
    print(f"  Files deleted (dupes):    {deleted}")
    print(f"  Files not found:          {not_found}")
    print(f"  Errors:                   {len(errors)}")
    print("=" * 60)
    
    if errors:
        print("\nERRORS:")
        for f, e in errors:
            print(f"  - {f}: {e}")
    
    # Self-destruct
    try:
        Path(__file__).unlink()
        print(f"\n[CLEANUP] Removed {Path(__file__).name}")
    except:
        pass
    
    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
