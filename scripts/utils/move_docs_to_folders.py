#!/usr/bin/env python3
"""
Move Documentation Files to Organized Folders
===============================================

Moves .md files from root to docs/ subdirectories.

Usage:
    python move_docs_to_folders.py --dry-run   # Preview
    python move_docs_to_folders.py --apply     # Execute
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def move_file(src: Path, dst: Path, dry_run: bool = True):
    if not src.exists():
        return False
    if dst.exists():
        log(f"⚠️  Exists (skipping): {dst}")
        return False
    
    action = "[DRY-RUN] Would move" if dry_run else "✅ Moving"
    log(f"{action}: {src.name} -> {dst.parent.name}/")
    
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True

def main():
    dry_run = "--apply" not in sys.argv
    
    log("=" * 60)
    log("Documentation Reorganization")
    log(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    log("=" * 60)
    
    root = Path(".")
    
    # Define moves: source -> destination
    moves = {
        # Architecture
        "ARCHITECTURE_RULES.md": "docs/architecture/RULES.md",
        "FEATURE_ARCHITECTURE_RULES.md": "docs/architecture/FEATURE_RULES.md",
        "MANAGEMENT_SYSTEMS.md": "docs/architecture/SYSTEMS.md",
        
        # Dashboard
        "DASHBOARD_README.md": "docs/dashboard/README.md",
        "MISSION_CONTROL_README.md": "docs/dashboard/MISSION_CONTROL.md",
        
        # Debugging
        "DEBUGGING_GUIDE.md": "docs/debugging/GUIDE.md",
        "DEBUGGING_OVERVIEW.md": "docs/debugging/OVERVIEW.md",
        "DEBUG_README.md": "docs/debugging/README.md",
        "PROJECT_DEBUGGING.md": "docs/debugging/PROJECT.md",
        "TROUBLESHOOTING_CHEATSHEET.md": "docs/debugging/TROUBLESHOOTING.md",
        "QUICK_FIX.md": "docs/debugging/QUICK_FIX.md",
        
        # Performance
        "PERFORMANCE_OPTIMIZATION.md": "docs/performance/OPTIMIZATION.md",
        "PERFORMANCE_SUMMARY.md": "docs/performance/SUMMARY.md",
        "OPTIMIZATION_REPORT.md": "docs/performance/REPORT.md",
        "OPTIMIZATIONS_APPLIED.md": "docs/performance/APPLIED.md",
        "OPTIMIZATIONS_BATCH_2.md": "docs/performance/BATCH_2.md",
        
        # Development
        "CLAUDE.md": "docs/development/CLAUDE.md",
        "COMPLETE_FEATURE_SUMMARY.md": "docs/development/FEATURE_SUMMARY.md",
        "LLM_ANALYSIS_REPORT.md": "docs/development/LLM_ANALYSIS.md",
        
        # Git Workflow
        "FINAL_COMMIT_GUIDE.md": "docs/git-workflow/COMMIT_GUIDE.md",
        "PUSH_TO_GITHUB.md": "docs/git-workflow/PUSH_TO_GITHUB.md",
        "🚀_COMMIT_AND_PUSH.md": "docs/git-workflow/QUICK_COMMIT.md",
        
        # History
        "DOCUMENTATION_UPDATE_v5.1.md": "docs/history/DOC_UPDATE_v5.1.md",
        "LLM_ROUTING_UPDATES_APPLIED.md": "docs/history/LLM_ROUTING_UPDATES.md",
        "ORGANIZATION_COMPLETE.md": "docs/history/ORGANIZATION.md",
        "UPDATE_SUMMARY.md": "docs/history/UPDATE_SUMMARY.md",
        "FINAL_SUMMARY.md": "docs/history/FINAL_SUMMARY.md",
        
        # Plugins
        "FEATURE_PROJECT_ANALYZER.md": "docs/plugins/PROJECT_ANALYZER.md",
        "docs_performance_PROJECT_ANALYZER.md": "docs/plugins/PROJECT_ANALYZER_PERF.md",
        
        # Maintenance
        "PROJECT_CLEANUP_GUIDE.md": "docs/maintenance/CLEANUP_GUIDE.md",
        
        # API
        "README_YAML_SPEC.md": "docs/api/YAML_SPEC.md",
        
        # Guides
        "USAGE_GUIDE.md": "docs/guides/USAGE_GUIDE.md",
        "CAPABILITIES.md": "docs/guides/CAPABILITIES.md",
    }
    
    # Files to delete (temporary/obsolete)
    to_delete = [
        "CLEANUP_SUGGESTIONS.md",
        "REORGANIZATION_PLAN.md",
        "SCRIPTS_README.md",
        "DOCS_REORGANIZATION.md",
    ]
    
    # Files to keep in root
    keep_in_root = [
        "README.md",
        "CONTRIBUTING.md",
    ]
    
    # Execute moves
    total = 0
    moved = 0
    
    log("\n📦 Moving documentation files...")
    for src_name, dst_path in moves.items():
        src = root / src_name
        dst = root / dst_path
        if move_file(src, dst, dry_run):
            moved += 1
        total += 1
    
    # Delete temporary files
    log("\n🗑️  Deleting temporary files...")
    deleted = 0
    for filename in to_delete:
        filepath = root / filename
        if filepath.exists():
            action = "[DRY-RUN] Would delete" if dry_run else "🗑️  Deleting"
            log(f"{action}: {filename}")
            if not dry_run:
                filepath.unlink()
                deleted += 1
        else:
            log(f"⚠️  Not found: {filename}")
    
    # Summary
    log(f"\n{'='*60}")
    log(f"Summary:")
    log(f"  Files to move: {total}")
    log(f"  Moved: {moved}")
    log(f"  Temp files deleted: {deleted}")
    log(f"  Keep in root: {len(keep_in_root)}")
    log(f"{'='*60}")
    
    if dry_run:
        log("\n⚠️  DRY RUN - No changes made. Use --apply to execute.")
    else:
        log("\n✅ Documentation reorganization complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
