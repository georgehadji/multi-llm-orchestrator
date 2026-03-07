"""Execute the move script."""
import shutil
from pathlib import Path

# Test files to move
test_files = [
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
]

root = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator")
tests_dir = root / "tests"
tests_dir.mkdir(exist_ok=True)

moved = []
skipped = []

for filename in test_files:
    src = root / filename
    dst = tests_dir / filename
    
    if not src.exists():
        skipped.append(f"{filename} (not found)")
        continue
    
    if dst.exists():
        # Remove source if destination exists (keep tests/ version)
        src.unlink()
        skipped.append(f"{filename} (already existed, removed duplicate)")
        continue
    
    shutil.move(str(src), str(dst))
    moved.append(filename)

print("=" * 60)
print("📁 Test Files Migration")
print("=" * 60)

if moved:
    print(f"\n✅ Moved ({len(moved)}):")
    for f in moved:
        print(f"   → tests/{f}")

if skipped:
    print(f"\n⏭️  Skipped/Handled ({len(skipped)}):")
    for msg in skipped:
        print(f"   • {msg}")

print("\n" + "=" * 60)
print("✅ Done!")
