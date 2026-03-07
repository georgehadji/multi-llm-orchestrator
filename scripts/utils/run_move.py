import shutil
from pathlib import Path

root = Path(r"E:\Documents\Vibe-Coding\Ai Orchestrator")
tests_dir = root / "tests"

files = [
    "test_ant_design_dashboard.py", "test_assembler_import.py", "test_deepseek.py",
    "test_enhanced_dashboard.py", "test_frontend_rules.py", "test_indesign_plugin_rules.py",
    "test_live_dashboard.py", "test_output_organizer.py", "test_performance_import.py",
    "test_wordpress_plugin_rules.py", "run_test.py", "check_frontend.py",
    "check_outputs.py", "check_state.py", "verify_frontend.py", "verify_syntax.py",
    "move_tests.py", "execute_move.py", "do_move.py"
]

for f in files:
    src, dst = root / f, tests_dir / f
    if src.exists():
        if dst.exists(): src.unlink()
        else: shutil.move(str(src), str(dst))
        print(f"✓ {f}")

print("\nDone!")
