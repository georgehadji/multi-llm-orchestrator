#!/usr/bin/env python3
"""
Root Folder Reorganization Script
==================================
Moves files from root to appropriate subdirectories.

Categories:
- scripts/setup/     : Setup and initialization scripts
- scripts/utils/     : Utility/check/debug scripts
- scripts/git/       : Git-related scripts
- scripts/batch/     : Windows batch files
- examples/          : Example usage scripts
- tests/             : Test files (already exists, move stragglers)
- docs/              : Documentation files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_file(src: Path, dst: Path, dry_run: bool = True):
    """Move file from src to dst."""
    if not src.exists():
        print(f"  ⚠️  Missing: {src}")
        return

    if dst.exists():
        print(f"  ⚠️  Exists: {dst} (skipping)")
        return

    action = "[DRY-RUN] Would move" if dry_run else "Moving"
    print(f"  {action}: {src.name} -> {dst}")

    if not dry_run:
        shutil.move(str(src), str(dst))


def main(dry_run: bool = True):
    root = Path(".")

    print(f"\n{'='*60}")
    print(f"Root Folder Reorganization")
    print(f"Mode: {'DRY-RUN (preview only)' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    # Define directories
    scripts_setup = ensure_dir(root / "scripts" / "setup")
    scripts_utils = ensure_dir(root / "scripts" / "utils")
    scripts_git = ensure_dir(root / "scripts" / "git")
    scripts_batch = ensure_dir(root / "scripts" / "batch")
    examples = ensure_dir(root / "examples")
    tests = ensure_dir(root / "tests")

    # Files to move

    # 1. SETUP SCRIPTS -> scripts/setup/
    setup_scripts = [
        "setup_project.py",
        "setup_scripts.py",
        "setup_github_workflow.py",
        "setup_slack_module.py",
        "init_project_structure.py",
        "create_all_files.py",
        "create_configs.py",
        "create_dashboard.py",
        "create_docs_dir.py",
        "create_github_templates.py",
        "create_github_workflow.py",
        "create_live_dashboard_script.py",
        "create_scripts.py",
        "create_scripts_folder.py",
        "create_slack_dir.py",
        "create_workflow.py",
        "temp_setup_slack.py",
    ]

    print(f"\n📦 Setup Scripts -> {scripts_setup}")
    for name in setup_scripts:
        move_file(root / name, scripts_setup / name, dry_run)

    # 2. UTILITY SCRIPTS -> scripts/utils/
    utility_scripts = [
        # Check scripts
        "check_all_syntax.py",
        "check_api_keys.py",
        "check_frontend.py",
        "check_outputs.py",
        "check_server.py",
        "check_state.py",
        "check_syntax.py",
        "check_syntax_mc.py",
        # Debug scripts
        "debug_dashboard.py",
        "debug_direct_import.py",
        "debug_import_step.py",
        "debug_mc_detailed.py",
        "debug_start.py",
        "diagnose_dashboard.py",
        "diagnose_mc.py",
        "diagnose_projects.py",
        # Cleanup scripts
        "cleanup_all.py",
        "cleanup_logging.py",
        "cleanup_temp.py",
        "cleanup_temp_files.py",
        "clear_cache.py",
        "final_cleanup.py",
        "final_cleanup_all.py",
        "remove_all_temp.py",
        # Fix scripts
        "fix_emojis.py",
        "fix_git_errors.py",
        "fix_logging.py",
        "fix_unicode.py",
        # Organization scripts
        "COMPLETE_REORGANIZATION.py",
        "_finalize_move.py",
        "do_move.py",
        "execute_move.py",
        "finalize_organization.py",
        "move_tests.py",
        "organize_codebase.py",
        "organize_docs.py",
        "perform_move.py",
        # Run scripts
        "run_dashboard.py",
        "run_dashboard_realtime.py",
        "run_mission_control_standalone.py",
        "run_move.py",
        "run_optimized_dashboard.py",
        "run_test.py",
        # Start scripts
        "start_and_test.py",
        "start_dashboard.py",
        "start_dashboard_debug.py",
        "start_dashboard_fixed.py",
        "start_mission_control.py",
        "start_simple.py",
        "start_simple_dashboard.py",
        # Kill/reset scripts
        "kill_server.py",
        "reset_dashboard.py",
        "restart_server.py",
        # Other
        "launcher.py",
        "temp_fix.py",
        "demo_live_dashboard.py",
        "quick_check.py",
        "quick_test.py",
        "test_runner.py",
        "update_doc_links.py",
        "verify_frontend.py",
        "verify_mc_import.py",
        "verify_syntax.py",
    ]

    print(f"\n🔧 Utility Scripts -> {scripts_utils}")
    for name in utility_scripts:
        move_file(root / name, scripts_utils / name, dry_run)

    # 3. GIT SCRIPTS -> scripts/git/
    git_scripts = [
        "clean_and_push.py",
        "commit_and_push.sh",
        "execute_git.py",
        "git_auto_commit.py",
        "git_commit_push.py",
    ]

    print(f"\n📁 Git Scripts -> {scripts_git}")
    for name in git_scripts:
        move_file(root / name, scripts_git / name, dry_run)

    # 4. BATCH FILES -> scripts/batch/
    batch_files = [
        "Check_API_Keys.bat",
        "COMMIT_COMMANDS.bat",
        "Debug_Dashboard.bat",
        "kill_port_8888.bat",
        "Mission_Control.bat",
        "restart_server.bat",
        "run_test.bat",
        "simple_test.bat",
        "start_dashboard.bat",
        "start_dashboard_antd.bat",
        "start_dashboard_live.bat",
        "Start_Mission_Control.bat",
        "start_server.bat",
        "start_with_log.bat",
        "stop_server.bat",
    ]

    print(f"\n🪟 Batch Files -> {scripts_batch}")
    for name in batch_files:
        move_file(root / name, scripts_batch / name, dry_run)

    # 5. EXAMPLES -> examples/
    example_files = [
        "example_capability_logging.py",
        "example_deepseek_coder.py",
        "example_deepseek_coder_v2.py",
        "example_enhanced_dashboard.py",
        "example_indesign_plugin_rules.py",
        "example_issue_tracking.py",
        "example_project_spec.yaml",
        "example_slack_integration.py",
        "example_wordpress_plugin_rules.py",
    ]

    print(f"\n📚 Examples -> {examples}")
    for name in example_files:
        move_file(root / name, examples / name, dry_run)

    # 6. TESTS -> tests/
    test_files = [
        "test_all_dashboards.py",
        "test_ant_design_dashboard.py",
        "test_api.py",
        "test_api_connection.py",
        "test_assembler_import.py",
        "test_dashboard_html.py",
        "test_dashboard_working.py",
        "test_deepseek.py",
        "test_direct_import.py",
        "test_enhanced_dashboard.py",
        "test_final_mc.py",
        "test_frontend_rules.py",
        "test_import.py",
        "test_import_debug.py",
        "test_indesign_plugin_rules.py",
        "test_live_dashboard.py",
        "test_mc_fixed.py",
        "test_mc_quick.py",
        "test_mc_simple.py",
        "test_mission_control.py",
        "test_mission_control_real.py",
        "test_output_organizer.py",
        "test_performance_import.py",
        "test_project.py",
        "test_quick.py",
        "test_remove_button.py",
        "test_server2.py",
        "test_simple_import.py",
        "test_startup.py",
        "test_v65_fix.py",
        "test_wordpress_plugin_rules.py",
    ]

    print(f"\n🧪 Tests -> {tests}")
    for name in test_files:
        move_file(root / name, tests / name, dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Setup scripts: {len(setup_scripts)}")
    print(f"Utility scripts: {len(utility_scripts)}")
    print(f"Git scripts: {len(git_scripts)}")
    print(f"Batch files: {len(batch_files)}")
    print(f"Examples: {len(example_files)}")
    print(f"Tests: {len(test_files)}")
    print(
        f"\nTotal files to move: {sum([len(setup_scripts), len(utility_scripts), len(git_scripts), len(batch_files), len(example_files), len(test_files)])}"
    )

    if dry_run:
        print(f"\n⚠️  This was a DRY RUN. No files were moved.")
        print(f"Run with --apply to execute the moves.")

    print()


if __name__ == "__main__":
    import sys

    dry_run = "--apply" not in sys.argv
    main(dry_run=dry_run)
