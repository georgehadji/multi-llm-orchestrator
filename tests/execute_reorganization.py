#!/usr/bin/env python3
"""
Execute Root Folder Reorganization
===================================

This script moves files from root to organized subdirectories.

Usage:
    python execute_reorganization.py --dry-run   # Preview changes
    python execute_reorganization.py --apply     # Execute moves
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def move_file(src: Path, dst: Path, dry_run: bool = True) -> bool:
    """Move file from src to dst."""
    if not src.exists():
        return False
    
    if dst.exists():
        log(f"⚠️  Exists (skipping): {dst.name}")
        return False
    
    action = "[DRY-RUN] Would move" if dry_run else "✅ Moving"
    log(f"{action}: {src.name} -> {dst.parent.name}/")
    
    if not dry_run:
        try:
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            log(f"❌ Error: {e}")
            return False
    return True

def main():
    dry_run = "--apply" not in sys.argv
    
    log("=" * 60)
    log("Root Folder Reorganization")
    log(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    log("=" * 60)
    
    root = Path(".")
    
    # Create directories
    dirs = {
        "setup": ensure_dir(root / "scripts" / "setup"),
        "utils": ensure_dir(root / "scripts" / "utils"),
        "git": ensure_dir(root / "scripts" / "git"),
        "batch": ensure_dir(root / "scripts" / "batch"),
        "examples": ensure_dir(root / "examples"),
    }
    
    categories = {
        "Setup Scripts": {
            "dest": dirs["setup"],
            "files": [
                "setup_project.py", "setup_scripts.py", "setup_github_workflow.py",
                "setup_slack_module.py", "init_project_structure.py", "create_all_files.py",
                "create_configs.py", "create_dashboard.py", "create_docs_dir.py",
                "create_github_templates.py", "create_github_workflow.py",
                "create_live_dashboard_script.py", "create_scripts.py",
                "create_scripts_folder.py", "create_slack_dir.py", "create_workflow.py",
                "temp_setup_slack.py",
            ]
        },
        "Git Scripts": {
            "dest": dirs["git"],
            "files": [
                "clean_and_push.py", "commit_and_push.sh", "execute_git.py",
                "git_auto_commit.py", "git_commit_push.py",
            ]
        },
        "Examples": {
            "dest": dirs["examples"],
            "files": [
                "example_capability_logging.py", "example_deepseek_coder.py",
                "example_deepseek_coder_v2.py", "example_enhanced_dashboard.py",
                "example_indesign_plugin_rules.py", "example_issue_tracking.py",
                "example_project_spec.yaml", "example_slack_integration.py",
                "example_wordpress_plugin_rules.py",
            ]
        },
    }
    
    # Utility scripts
    utils_files = [
        "check_all_syntax.py", "check_api_keys.py", "check_frontend.py",
        "check_outputs.py", "check_server.py", "check_state.py",
        "check_syntax.py", "check_syntax_mc.py", "debug_dashboard.py",
        "debug_direct_import.py", "debug_import_step.py", "debug_mc_detailed.py",
        "debug_start.py", "diagnose_dashboard.py", "diagnose_mc.py",
        "diagnose_projects.py", "cleanup_all.py", "cleanup_logging.py",
        "cleanup_temp.py", "cleanup_temp_files.py", "clear_cache.py",
        "final_cleanup.py", "final_cleanup_all.py", "remove_all_temp.py",
        "fix_emojis.py", "fix_git_errors.py", "fix_logging.py", "fix_unicode.py",
        "COMPLETE_REORGANIZATION.py", "_finalize_move.py", "do_move.py",
        "execute_move.py", "finalize_organization.py", "move_tests.py",
        "organize_codebase.py", "organize_docs.py", "perform_move.py",
        "temp_fix.py", "run_dashboard.py", "run_dashboard_realtime.py",
        "run_mission_control_standalone.py", "run_move.py", "run_optimized_dashboard.py",
        "run_test.py", "start_and_test.py", "start_dashboard.py",
        "start_dashboard_debug.py", "start_dashboard_fixed.py", "start_mission_control.py",
        "start_simple.py", "start_simple_dashboard.py", "kill_server.py",
        "reset_dashboard.py", "restart_server.py", "launcher.py",
        "demo_live_dashboard.py", "quick_check.py", "quick_test.py",
        "test_runner.py", "update_doc_links.py", "verify_frontend.py",
        "verify_mc_import.py", "verify_syntax.py",
    ]
    
    categories["Utility Scripts"] = {
        "dest": dirs["utils"],
        "files": utils_files
    }
    
    # Batch files
    categories["Batch Files"] = {
        "dest": dirs["batch"],
        "files": [
            "Check_API_Keys.bat", "COMMIT_COMMANDS.bat", "Debug_Dashboard.bat",
            "kill_port_8888.bat", "Mission_Control.bat", "restart_server.bat",
            "run_test.bat", "simple_test.bat", "start_dashboard.bat",
            "start_dashboard_antd.bat", "start_dashboard_live.bat",
            "Start_Mission_Control.bat", "start_server.bat", "start_with_log.bat",
            "stop_server.bat",
        ]
    }
    
    # Execute moves
    total = 0
    moved = 0
    
    for category_name, category in categories.items():
        log(f"\n📦 {category_name} -> {category['dest'].name}/")
        for filename in category["files"]:
            src = root / filename
            dst = category["dest"] / filename
            if move_file(src, dst, dry_run):
                moved += 1
            total += 1
    
    log(f"\n{'='*60}")
    log(f"Summary: {moved}/{total} files")
    log(f"{'='*60}")
    
    if dry_run:
        log("\n⚠️ DRY RUN - No files moved. Use --apply to execute.")
    else:
        log("\n✅ Done!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
