#!/usr/bin/env python3
"""
Sync D: drive (source of truth) to E: drive
Ensures both AI Orchestrator installations are identical
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def copy_file(src: Path, dst: Path) -> bool:
    """Copy a file if source is newer or destination doesn't exist."""
    if not src.exists():
        return False
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if update needed
    if dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        if src_stat.st_mtime <= dst_stat.st_mtime and src_stat.st_size == dst_stat.st_size:
            return False  # No update needed
    
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        log(f"❌ Error copying {src.name}: {e}")
        return False

def main():
    log("=" * 60)
    log("AI Orchestrator Sync: D: -> E:")
    log("=" * 60)
    
    d_root = Path("D:/Vibe-Coding/Ai Orchestrator")
    e_root = Path("E:/Documents/Vibe-Coding/Ai Orchestrator")
    
    if not d_root.exists():
        log("❌ D: drive source not found!")
        return 1
    
    if not e_root.exists():
        log(f"❌ E: drive destination not found!")
        return 1
    
    # Define sync patterns
    sync_patterns = [
        # (source_pattern, dest_dir, description)
        ("orchestrator/*.py", "orchestrator", "Core Python modules"),
        ("docs/**/*.md", "docs", "Documentation"),
        ("tests/**/*.py", "tests", "Tests"),
        ("examples/*", "examples", "Examples"),
        ("scripts/**/*", "scripts", "Utility scripts"),
    ]
    
    # Individual files in root
    root_files = [
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        "README.md",
        "CONTRIBUTING.md",
        "Makefile",
        "Dockerfile",
        "CLAUDE.md",
    ]
    
    total_copied = 0
    total_skipped = 0
    
    # Sync directory patterns
    for pattern, dest_subdir, desc in sync_patterns:
        log(f"\n📦 Syncing {desc}...")
        files = list(d_root.glob(pattern))
        copied = 0
        skipped = 0
        
        for src_file in files:
            if src_file.is_dir():
                continue
            rel_path = src_file.relative_to(d_root)
            dst_file = e_root / rel_path
            
            if copy_file(src_file, dst_file):
                copied += 1
            else:
                skipped += 1
        
        log(f"  Copied: {copied}, Skipped: {skipped}")
        total_copied += copied
        total_skipped += skipped
    
    # Sync root files
    log(f"\n📄 Syncing root config files...")
    for filename in root_files:
        src = d_root / filename
        dst = e_root / filename
        if src.exists():
            if copy_file(src, dst):
                log(f"  ✅ Copied: {filename}")
                total_copied += 1
            else:
                total_skipped += 1
    
    # Summary
    log(f"\n{'='*60}")
    log(f"Sync Complete!")
    log(f"  Files copied: {total_copied}")
    log(f"  Files skipped (up to date): {total_skipped}")
    log(f"{'='*60}")
    
    # List critical new files
    log("\n🔍 Verifying key new files:")
    new_files = [
        "orchestrator/issue_tracking.py",
        "orchestrator/slack_integration.py",
        "orchestrator/git_service.py",
        "orchestrator/git_hooks.py",
        "docs/ISSUE_TRACKING.md",
    ]
    
    for file_path in new_files:
        full_path = e_root / file_path
        status = "✅" if full_path.exists() else "❌ MISSING"
        log(f"  {status} {file_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
