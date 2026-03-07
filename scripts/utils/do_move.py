"""Move test files from root to tests/ folder."""
import shutil
import os
from pathlib import Path

root = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator")
tests_dir = root / "tests"

files_to_move = [
    "test_ant_design_dashboard.py",
    "test_assembler_import.py", 
    "test_deepseek.py",
    "test_enhanced_dashboard.py",
    "test_frontend_rules.py",
    "test_indesign_plugin_rules.py",
    "test_live_dashboard.py",
    "test_output_organizer.py",
    "test_performance_import.py",
    "test_wordpress_plugin_rules.py",
    "run_test.py",
    "check_frontend.py",
    "check_outputs.py",
    "check_state.py",
    "verify_frontend.py",
    "verify_syntax.py",
    "move_tests.py",
    "execute_move.py",
]

print("📁 Moving test files to tests/ folder...\n")

moved = []
not_found = []
errors = []

for filename in files_to_move:
    src = root / filename
    dst = tests_dir / filename
    
    if not src.exists():
        not_found.append(filename)
        continue
    
    try:
        # If destination exists, delete source (it's a duplicate)
        if dst.exists():
            src.unlink()
            print(f"  🗑️  Removed duplicate: {filename}")
        else:
            shutil.move(str(src), str(dst))
            print(f"  ✅ Moved: {filename}")
        moved.append(filename)
    except Exception as e:
        print(f"  ❌ Error with {filename}: {e}")
        errors.append((filename, str(e)))

print(f"\n{'='*50}")
print(f"Summary: {len(moved)} processed, {len(not_found)} not found, {len(errors)} errors")

if errors:
    print("\nErrors:")
    for f, e in errors:
        print(f"  {f}: {e}")
