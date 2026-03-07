#!/usr/bin/env python3
"""Move test files to tests/ folder."""
import shutil
from pathlib import Path

# Test files to move
test_files = [
    # test_*.py files
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
    # Other test-related files
    "run_test.py",
    "check_frontend.py",
    "check_outputs.py",
    "check_state.py",
    "verify_frontend.py",
    "verify_syntax.py",
]

root = Path(".")
tests_dir = root / "tests"

# Ensure tests directory exists
tests_dir.mkdir(exist_ok=True)

moved = []
skipped = []
errors = []

for filename in test_files:
    src = root / filename
    dst = tests_dir / filename
    
    if not src.exists():
        skipped.append(f"{filename} (not found)")
        continue
    
    if dst.exists():
        skipped.append(f"{filename} (already exists in tests/)")
        continue
    
    try:
        shutil.move(str(src), str(dst))
        moved.append(filename)
    except Exception as e:
        errors.append(f"{filename}: {e}")

# Print results
print("=" * 60)
print("📁 Test Files Migration Report")
print("=" * 60)

if moved:
    print(f"\n✅ Moved ({len(moved)} files):")
    for f in moved:
        print(f"   → tests/{f}")

if skipped:
    print(f"\n⏭️  Skipped ({len(skipped)}):")
    for msg in skipped:
        print(f"   • {msg}")

if errors:
    print(f"\n❌ Errors ({len(errors)}):")
    for msg in errors:
        print(f"   • {msg}")

if not moved and not skipped and not errors:
    print("\nℹ️  No files to move")

print("\n" + "=" * 60)
print("Done!")
