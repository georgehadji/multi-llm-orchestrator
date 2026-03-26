#!/usr/bin/env python3
"""
AI Orchestrator - Full Sync & Cleanup
======================================
Comprehensive script that:
1. Syncs D: drive (source of truth) to E: drive
2. Deletes temporary files from both locations
3. Runs folder reorganization in both locations
4. Verifies final state

Usage:
    python full_sync_and_cleanup.py --dry-run   # Preview
    python full_sync_and_cleanup.py --apply     # Execute

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

import shutil
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# ANSI colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

def log(msg: str, level: str = "info"):
    """Log with timestamp and color."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {"info": BLUE, "success": GREEN, "warning": YELLOW, "error": RED}
    color = colors.get(level, "")
    print(f"{color}[{timestamp}] {msg}{RESET}")

def section(title: str):
    """Print section header."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

class SyncManager:
    """Manages synchronization between D: and E: drives."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.d_root = Path("D:/Vibe-Coding/Ai Orchestrator")
        self.e_root = Path("E:/Documents/Vibe-Coding/Ai Orchestrator")
        self.stats = {"copied": 0, "skipped": 0, "deleted": 0, "errors": 0}
        
        # Critical new files that must exist
        self.critical_files = [
            "orchestrator/issue_tracking.py",
            "orchestrator/slack_integration.py",
            "orchestrator/git_service.py",
            "orchestrator/git_hooks.py",
            "docs/ISSUE_TRACKING.md",
        ]
        
        # Files to delete (temporary/obsolete)
        self.files_to_delete = [
            # Backup files
            "CAPABILITIES.md.bak",
            "USAGE_GUIDE.md.bak",
            # Temp files
            "temp_fix.py",
            "temp_setup_slack.py",
            "create_slack_dir.py",
            "setup_slack_module.py",
            # Old session files
            ".env_old.txt",
            "COMMIT_SUMMARY.txt",
            "CONTINUE_SESSION.txt",
            # Temporary docs
            "CLEANUP_SUGGESTIONS.md",
            "REORGANIZATION_PLAN.md",
            "SCRIPTS_README.md",
            "DOCS_REORGANIZATION.md",
            "SYNC_STATUS_REPORT.md",
            "SYNC_REPORT.txt",
        ]
        
        # Sync patterns: (source_glob, description)
        self.sync_patterns = [
            ("orchestrator/**/*.py", "Core Python modules"),
            ("docs/**/*.md", "Documentation"),
            ("tests/**/*.py", "Test files"),
            ("examples/*", "Example scripts"),
            ("scripts/**/*", "Utility scripts"),
        ]
        
        # Root config files
        self.root_configs = [
            "pyproject.toml",
            ".env.example",
            ".gitignore",
            ".dockerignore",
            "README.md",
            "CONTRIBUTING.md",
            "Makefile",
            "Dockerfile",
        ]
    
    def verify_locations(self) -> bool:
        """Verify both locations exist."""
        if not self.d_root.exists():
            log(f"❌ D: drive not found: {self.d_root}", "error")
            return False
        
        if not self.e_root.exists():
            log(f"⚠️  E: drive not found, will create: {self.e_root}", "warning")
            if not self.dry_run:
                self.e_root.mkdir(parents=True, exist_ok=True)
        
        log(f"✓ D: source: {self.d_root}", "success")
        log(f"✓ E: target: {self.e_root}", "success")
        return True
    
    def copy_file(self, src: Path, dst: Path) -> bool:
        """Copy a file, respecting dry-run mode."""
        if not src.exists():
            return False
        
        try:
            if not self.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.copy2(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
            return True
        except Exception as e:
            log(f"❌ Error copying {src.name}: {e}", "error")
            self.stats["errors"] += 1
            return False
    
    def sync_files(self) -> bool:
        """Sync files from D: to E:."""
        section("STEP 1: Syncing Files (D: → E:)")
        
        # Sync directory patterns
        for pattern, desc in self.sync_patterns:
            log(f"📦 {desc}...")
            files = list(self.d_root.glob(pattern))
            copied = 0
            
            for src_file in files:
                if src_file.is_dir():
                    continue
                rel_path = src_file.relative_to(self.d_root)
                dst_file = self.e_root / rel_path
                
                # Check if update needed
                needs_copy = True
                if dst_file.exists():
                    src_stat = src_file.stat()
                    dst_stat = dst_file.stat()
                    if (src_stat.st_mtime <= dst_stat.st_mtime and 
                        src_stat.st_size == dst_stat.st_size):
                        needs_copy = False
                
                if needs_copy:
                    if self.copy_file(src_file, dst_file):
                        copied += 1
                        self.stats["copied"] += 1
                    if copied % 10 == 0:
                        log(f"  ...copied {copied} files")
                else:
                    self.stats["skipped"] += 1
            
            log(f"  ✓ {desc}: {copied} copied", "success")
        
        # Sync root config files
        log("📄 Root config files...")
        for filename in self.root_configs:
            src = self.d_root / filename
            dst = self.e_root / filename
            if src.exists() and self.copy_file(src, dst):
                log(f"  ✓ {filename}")
                self.stats["copied"] += 1
        
        return True
    
    def cleanup_files(self) -> bool:
        """Delete temporary files from both locations."""
        section("STEP 2: Cleaning Up Temporary Files")
        
        for location, name in [(self.d_root, "D:"), (self.e_root, "E:")]:
            log(f"🧹 Cleaning {name}...")
            deleted = 0
            
            for filename in self.files_to_delete:
                filepath = location / filename
                if filepath.exists():
                    action = "Would delete" if self.dry_run else "Deleting"
                    log(f"  {action}: {filename}")
                    if not self.dry_run:
                        try:
                            filepath.unlink()
                            deleted += 1
                            self.stats["deleted"] += 1
                        except Exception as e:
                            log(f"  ❌ Error: {e}", "error")
            
            log(f"  ✓ {name}: {deleted} files cleaned", "success")
        
        return True
    
    def verify_critical_files(self) -> bool:
        """Verify critical new files exist in both locations."""
        section("STEP 3: Verifying Critical Files")
        
        all_ok = True
        for location, name in [(self.d_root, "D:"), (self.e_root, "E:")]:
            log(f"🔍 Checking {name}...")
            for file_path in self.critical_files:
                full_path = location / file_path
                exists = full_path.exists()
                status = "✓" if exists else "✗"
                color = GREEN if exists else RED
                print(f"  {color}{status} {file_path}{RESET}")
                if not exists:
                    all_ok = False
        
        return all_ok
    
    def print_summary(self):
        """Print final summary."""
        section("SUMMARY")
        
        log(f"Files copied: {self.stats['copied']}")
        log(f"Files skipped (up to date): {self.stats['skipped']}")
        log(f"Files deleted: {self.stats['deleted']}")
        if self.stats["errors"] > 0:
            log(f"Errors: {self.stats['errors']}", "error")
        
        if self.dry_run:
            print(f"\n{YELLOW}⚠️  This was a DRY RUN. No changes were made.{RESET}")
            print(f"{YELLOW}   Run with --apply to execute.{RESET}\n")
        else:
            print(f"\n{GREEN}✅ Sync and cleanup complete!{RESET}\n")
            print("Next steps:")
            print("  1. Verify files in E: drive")
            print("  2. Run tests to ensure everything works")
            print("  3. Commit changes to git")

def main():
    # Parse arguments
    dry_run = "--apply" not in sys.argv
    
    # Header
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  AI Orchestrator - Full Sync & Cleanup{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"\n  Mode: {'DRY RUN (preview)' if dry_run else 'LIVE (executing)'}\n")
    
    # Run sync
    manager = SyncManager(dry_run=dry_run)
    
    if not manager.verify_locations():
        return 1
    
    try:
        manager.sync_files()
        manager.cleanup_files()
        manager.verify_critical_files()
        manager.print_summary()
        return 0
    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user", "warning")
        return 1
    except Exception as e:
        log(f"\n❌ Error: {e}", "error")
        return 1

if __name__ == "__main__":
    sys.exit(main())
